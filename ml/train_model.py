import pandas as pd
import numpy as np
import psycopg2
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.decomposition import PCA
import tempfile
import os

# 统一配色
COLORS = {"setosa": "#2ecc71", "versicolor": "#3498db", "virginica": "#e74c3c"}
sns.set_theme(style="whitegrid", font_scale=1.1)

# 1. 连接数据库
conn = psycopg2.connect(
    host="db", port=5432,
    user=os.environ.get("POSTGRES_USER", "myuser"),
    password=os.environ.get("POSTGRES_PASSWORD", "mypass"),
    dbname=os.environ.get("POSTGRES_DB", "mydb")
)
df = pd.read_sql("SELECT * FROM fct_measurements", conn)
conn.close()
print(f"Loaded {len(df)} rows from fct_measurements")

# 2. 准备数据
feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width",
                "sepal_area", "petal_area"]
X = df[feature_cols]
y = df["species"]
species_labels = sorted(y.unique())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 连接 MLflow
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("iris_classification")

# 4. 训练 + 记录
with mlflow.start_run():
    n_estimators = 100
    max_depth = 5

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    # 5-折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accuracy = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    cv_f1 = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted")

    # 记录参数
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("features", feature_cols)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))

    # 记录指标
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("cv_accuracy_mean", cv_accuracy.mean())
    mlflow.log_metric("cv_accuracy_std", cv_accuracy.std())
    mlflow.log_metric("cv_f1_mean", cv_f1.mean())
    mlflow.log_metric("cv_f1_std", cv_f1.std())
    for i, (acc, f) in enumerate(zip(cv_accuracy, cv_f1)):
        mlflow.log_metric(f"cv_fold{i+1}_accuracy", acc)
        mlflow.log_metric(f"cv_fold{i+1}_f1", f)
    for name, imp in zip(feature_cols, model.feature_importances_):
        mlflow.log_metric(f"importance_{name}", round(imp, 4))

    # === 生成图表 ===
    tmp_dir = tempfile.mkdtemp()

    # 图1: 混淆矩阵热力图（带百分比）
    cm = confusion_matrix(y_test, y_pred, labels=species_labels)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=2, linecolor="white",
                xticklabels=species_labels, yticklabels=species_labels, ax=ax,
                cbar_kws={"label": "Count"}, square=True)
    # 在数字下面加百分比
    for i in range(len(species_labels)):
        for j in range(len(species_labels)):
            ax.text(j + 0.5, i + 0.7, f"({cm_pct[i, j]:.0f}%)",
                    ha="center", va="center", fontsize=9, color="gray")
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(tmp_dir, "1_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 图2: ROC 曲线（每个品种一条线）
    y_test_bin = label_binarize(y_test, classes=species_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, species in enumerate(species_labels):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[species], linewidth=2.5,
                label=f"{species} (AUC = {roc_auc:.2f})")
        ax.fill_between(fpr, tpr, alpha=0.1, color=COLORS[species])
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC = 0.50)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(os.path.join(tmp_dir, "2_roc_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 图3: 特征重要性（水平圆点图 lollipop chart）
    importances = model.feature_importances_
    indices = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(feature_cols))
    colors = plt.cm.viridis(importances[indices] / importances.max())
    ax.hlines(y_pos, 0, importances[indices], color=colors, linewidth=2.5)
    ax.scatter(importances[indices], y_pos, color=colors, s=120, zorder=3, edgecolors="white", linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_cols[i] for i in indices], fontsize=11)
    for i, v in enumerate(importances[indices]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=10)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("Feature Importance", fontsize=14, fontweight="bold")
    ax.set_xlim(0, importances.max() * 1.2)
    sns.despine(left=True)
    plt.tight_layout()
    fig.savefig(os.path.join(tmp_dir, "3_feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 图4: PCA 2D 散点图（降维可视化）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(8, 6))
    for species in species_labels:
        mask = y == species
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=COLORS[species],
                   label=species, s=60, alpha=0.7, edgecolors="white", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
    ax.set_title("PCA — 2D Projection of Iris Data", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    sns.despine()
    plt.tight_layout()
    fig.savefig(os.path.join(tmp_dir, "4_pca_scatter.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 图5: 特征分布小提琴图
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(feature_cols):
        sns.violinplot(data=df, x="species", y=col, hue="species",
                       palette=COLORS, ax=axes[i], inner="quart", linewidth=1.2, legend=False)
        axes[i].set_title(col.replace("_", " ").title(), fontsize=12, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")
    fig.suptitle("Feature Distribution by Species", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(tmp_dir, "5_feature_violin.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 图6: 交叉验证雷达图
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    acc_vals = cv_accuracy.tolist() + [cv_accuracy[0]]
    f1_vals = cv_f1.tolist() + [cv_f1[0]]
    ax.plot(angles, acc_vals, "o-", linewidth=2, color="#3498db", label=f"Accuracy (μ={cv_accuracy.mean():.3f})")
    ax.fill(angles, acc_vals, alpha=0.15, color="#3498db")
    ax.plot(angles, f1_vals, "s-", linewidth=2, color="#e74c3c", label=f"F1 Score (μ={cv_f1.mean():.3f})")
    ax.fill(angles, f1_vals, alpha=0.15, color="#e74c3c")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"Fold {i+1}" for i in range(5)], fontsize=11)
    ax.set_ylim(0.85, 1.02)
    ax.set_title("5-Fold Cross Validation", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.2, -0.05), fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(tmp_dir, "6_cv_radar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 上传图表
    mlflow.log_artifacts(tmp_dir, "charts")

    # 记录模型
    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"Model trained: accuracy={accuracy:.4f}, f1={f1:.4f}")
    print(f"Cross-validation: accuracy={cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
    print("6 charts + model logged to MLflow")
