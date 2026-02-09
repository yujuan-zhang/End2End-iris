import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             accuracy_score, f1_score, precision_score, recall_score)
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="Iris Dashboard", layout="wide")

COLORS = {"setosa": "#2ecc71", "versicolor": "#3498db", "virginica": "#e74c3c"}
COLOR_LIST = ["#2ecc71", "#3498db", "#e74c3c"]


@st.cache_data
def load_data():
    """Load data from PostgreSQL (Docker) or CSV (Streamlit Cloud)."""
    # Try database first (Docker environment)
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="db", port=5432,
            user=os.environ.get("POSTGRES_USER", "myuser"),
            password=os.environ.get("POSTGRES_PASSWORD", "mypass"),
            dbname=os.environ.get("POSTGRES_DB", "mydb")
        )
        df = pd.read_sql("SELECT * FROM fct_measurements", conn)
        df_summary = pd.read_sql("SELECT * FROM mart_species_summary", conn)
        conn.close()
        return df, df_summary
    except Exception:
        pass

    # Fallback: read from CSV (Streamlit Cloud)
    csv_path = Path(__file__).parent.parent / "data" / "iris.csv"
    raw = pd.read_csv(csv_path)

    # Replicate dbt staging: round + size_category
    for col in ["sepal_length", "sepal_width", "petal_length", "petal_width"]:
        raw[col] = raw[col].round(1)
    raw["size_category"] = pd.cut(
        raw["sepal_length"],
        bins=[0, 5.0, 6.5, float("inf")],
        labels=["small", "medium", "large"],
        right=False
    )
    raw["measurement_id"] = range(1, len(raw) + 1)

    # Replicate dbt fct_measurements: add computed fields
    raw["sepal_area"] = (raw["sepal_length"] * raw["sepal_width"]).round(1)
    raw["petal_area"] = (raw["petal_length"] * raw["petal_width"]).round(1)
    df = raw

    # Replicate dbt mart_species_summary
    df_summary = (
        df.groupby(["species", "size_category"], observed=True)
        .agg(
            total=("measurement_id", "count"),
            avg_sepal_length=("sepal_length", "mean"),
            avg_sepal_width=("sepal_width", "mean"),
            avg_petal_length=("petal_length", "mean"),
            avg_petal_width=("petal_width", "mean"),
            avg_sepal_area=("sepal_area", "mean"),
            avg_petal_area=("petal_area", "mean"),
        )
        .round(1)
        .reset_index()
        .sort_values(["species", "size_category"])
    )

    return df, df_summary


try:
    df, df_summary = load_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width",
                "sepal_area", "petal_area"]
species_list = sorted(df["species"].unique().tolist())

# ========== 页面标题 ==========
st.title("Iris Data Pipeline Dashboard")
st.markdown("""
End-to-end data pipeline built with **Docker**: PostgreSQL + dbt + Airflow + MLflow + Python + Streamlit.

This dashboard is the visualization layer of the pipeline, showing interactive data exploration
and machine learning analysis on the classic [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set)
(150 samples, 3 species, 4 measurements).

| Component | Role |
|---|---|
| **PostgreSQL** | Data storage |
| **dbt** | Data transformation (staging → marts) + testing |
| **Airflow** | Pipeline orchestration |
| **Python** | Static reports (CSV + PNG) |
| **MLflow** | ML experiment tracking |
| **Streamlit** | Interactive dashboard (this page) |
""")

st.divider()

# ========== 页面导航 ==========
tab1, tab2 = st.tabs(["Data Explorer", "ML Analysis"])

# ========== Tab 1: 数据探索 ==========
with tab1:

    st.header("1. Summary Table")
    st.dataframe(df_summary, use_container_width=True)

    st.header("2. Scatter Plot")
    col1, col2 = st.columns(2)
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    with col1:
        x_axis = st.selectbox("X Axis", features, index=0)
    with col2:
        y_axis = st.selectbox("Y Axis", features, index=1)

    selected_species = st.multiselect("Filter by Species", species_list, default=species_list)
    filtered_df = df[df["species"].isin(selected_species)]

    fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color="species",
                     title=f"{x_axis} vs {y_axis}", color_discrete_map=COLORS)
    st.plotly_chart(fig, use_container_width=True)

    st.header("3. Average Measurements by Species")
    avg_df = filtered_df.groupby("species")[features].mean().reset_index()
    avg_melted = avg_df.melt(id_vars="species", var_name="measurement", value_name="average")
    fig2 = px.bar(avg_melted, x="species", y="average", color="measurement",
                  barmode="group", title="Average Measurements")
    st.plotly_chart(fig2, use_container_width=True)

    st.header("4. Statistics")
    col1, col2, col3 = st.columns(3)
    for i, species in enumerate(species_list):
        with [col1, col2, col3][i]:
            species_data = df[df["species"] == species]
            st.subheader(species)
            st.metric("Count", len(species_data))
            st.metric("Avg Sepal Length", round(species_data["sepal_length"].mean(), 2))
            st.metric("Avg Petal Length", round(species_data["petal_length"].mean(), 2))

# ========== Tab 2: ML 分析 ==========
with tab2:
    st.title("ML Analysis — Random Forest Classifier")

    @st.cache_data
    def train_model():
        X = df[feature_cols]
        y = df["species"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        cv_f1 = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted")
        return model, X_train, X_test, y_train, y_test, y_pred, y_proba, cv_acc, cv_f1

    model, X_train, X_test, y_train, y_test, y_pred, y_proba, cv_acc, cv_f1 = train_model()

    # --- 指标卡片 ---
    st.header("1. Model Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
    m2.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.2%}")
    m3.metric("CV Accuracy", f"{cv_acc.mean():.2%} ± {cv_acc.std():.2%}")
    m4.metric("CV F1", f"{cv_f1.mean():.2%} ± {cv_f1.std():.2%}")

    # --- 混淆矩阵 + ROC ---
    st.header("2. Confusion Matrix & ROC Curves")
    col_left, col_right = st.columns(2)

    with col_left:
        cm = confusion_matrix(y_test, y_pred, labels=species_list)
        cm_text = [[str(v) for v in row] for row in cm]
        fig_cm = ff.create_annotated_heatmap(
            cm, x=species_list, y=species_list,
            annotation_text=cm_text, colorscale="Blues", showscale=True
        )
        fig_cm.update_layout(title="Confusion Matrix",
                             xaxis_title="Predicted", yaxis_title="Actual",
                             yaxis=dict(autorange="reversed"), height=420)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_right:
        y_test_bin = label_binarize(y_test, classes=species_list)
        fig_roc = go.Figure()
        for i, species in enumerate(species_list):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            r, g, b = int(COLOR_LIST[i][1:3], 16), int(COLOR_LIST[i][3:5], 16), int(COLOR_LIST[i][5:7], 16)
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, name=f"{species} (AUC={roc_auc:.2f})",
                fill="tozeroy", line=dict(color=COLOR_LIST[i], width=2),
                fillcolor=f"rgba({r},{g},{b},0.1)"
            ))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random",
                                     line=dict(dash="dash", color="gray")))
        fig_roc.update_layout(title="ROC Curves (One-vs-Rest)",
                              xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate", height=420)
        st.plotly_chart(fig_roc, use_container_width=True)

    # --- 特征重要性 + PCA ---
    st.header("3. Feature Importance & PCA")
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        importances = model.feature_importances_
        indices = np.argsort(importances)
        fig_imp = go.Figure(go.Bar(
            x=importances[indices],
            y=[feature_cols[i] for i in indices],
            orientation="h",
            marker=dict(color=importances[indices], colorscale="Viridis"),
            text=[f"{v:.3f}" for v in importances[indices]],
            textposition="outside"
        ))
        fig_imp.update_layout(title="Feature Importance", xaxis_title="Importance",
                              height=420, xaxis=dict(range=[0, importances.max() * 1.25]))
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_right2:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[feature_cols])
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "species": df["species"]})
        fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="species",
                             color_discrete_map=COLORS,
                             title=f"PCA (PC1: {pca.explained_variance_ratio_[0]:.1%}, PC2: {pca.explained_variance_ratio_[1]:.1%})")
        fig_pca.update_layout(height=420)
        st.plotly_chart(fig_pca, use_container_width=True)

    # --- 小提琴图 ---
    st.header("4. Feature Distribution by Species")
    col_a, col_b = st.columns(2)
    for i, col in enumerate(feature_cols):
        target_col = col_a if i % 2 == 0 else col_b
        with target_col:
            fig_v = px.violin(df, x="species", y=col, color="species",
                              color_discrete_map=COLORS, box=True, points="all",
                              title=col.replace("_", " ").title())
            fig_v.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_v, use_container_width=True)

    # --- 交叉验证 ---
    st.header("5. Cross Validation (5-Fold)")
    cv_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(5)] * 2,
        "Score": np.concatenate([cv_acc, cv_f1]),
        "Metric": ["Accuracy"] * 5 + ["F1 Score"] * 5
    })
    fig_cv = px.bar(cv_df, x="Fold", y="Score", color="Metric", barmode="group",
                    color_discrete_map={"Accuracy": "#3498db", "F1 Score": "#e74c3c"},
                    title=f"5-Fold CV — Accuracy: {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
    fig_cv.update_layout(yaxis=dict(range=[0.85, 1.02]), height=400)
    fig_cv.add_hline(y=cv_acc.mean(), line_dash="dash", line_color="#3498db", opacity=0.5)
    fig_cv.add_hline(y=cv_f1.mean(), line_dash="dash", line_color="#e74c3c", opacity=0.5)
    st.plotly_chart(fig_cv, use_container_width=True)
