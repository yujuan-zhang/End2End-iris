import pandas as pd
import os
import psycopg2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 连接数据库
conn = psycopg2.connect(
    host="db", port=5432,
    user=os.environ.get("POSTGRES_USER", "myuser"),
    password=os.environ.get("POSTGRES_PASSWORD", "mypass"),
    dbname=os.environ.get("POSTGRES_DB", "mydb")
)

# 1. 导出 iris_summary 为 CSV 报告
df_summary = pd.read_sql("SELECT * FROM mart_species_summary", conn)
df_summary.to_csv("/results/iris_summary.csv", index=False)
print("CSV 报告已生成: results/iris_summary.csv")

# 2. 读取原始数据用于画图
df = pd.read_sql("SELECT * FROM fct_measurements", conn)
conn.close()

# 3. 柱状图：各品种特征平均值对比
fig, ax = plt.subplots(figsize=(10, 6))
features = ["avg_sepal_length", "avg_sepal_width", "avg_petal_length", "avg_petal_width"]
labels = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]

# 按品种汇总（不含 size_category）
species_avg = df.groupby("species")[["sepal_length", "sepal_width", "petal_length", "petal_width"]].mean()
species_avg.plot(kind="bar", ax=ax)
ax.set_title("Average Measurements by Species")
ax.set_ylabel("cm")
ax.set_xlabel("Species")
ax.legend(["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("/results/bar_chart.png", dpi=150)
print("柱状图已生成: results/bar_chart.png")

# 4. 散点图：花萼长度 vs 宽度，按品种着色
fig, ax = plt.subplots(figsize=(8, 6))
colors = {"setosa": "#e74c3c", "versicolor": "#2ecc71", "virginica": "#3498db"}
for species, group in df.groupby("species"):
    ax.scatter(group["sepal_length"], group["sepal_width"],
               label=species, color=colors[species], alpha=0.7)
ax.set_title("Sepal Length vs Width by Species")
ax.set_xlabel("Sepal Length (cm)")
ax.set_ylabel("Sepal Width (cm)")
ax.legend()
plt.tight_layout()
plt.savefig("/results/scatter_plot.png", dpi=150)
print("散点图已生成: results/scatter_plot.png")

print("全部完成!")
