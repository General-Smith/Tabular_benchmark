import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import numpy as np

# ===================== 全局绘图配置（统一所有图）=====================
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# ===================== 统一配色 =====================
color_map = {
    # 树模型（绿色系）
    "RandomForest": "#2E7D32",
    "XGBoost": "#388E3C",
    "LightGBM": "#43A047",
    "CatBoost": "#4CAF50",
    # 深度学习（橙红色系）
    "MLP": "#E64A19",
    "ResNet": "#F57C00",
    "FT-Transformer": "#FF9800",
    "TabNet": "#FFB74D"
}

# ===================== 读取结果（只读一次）=====================
with open("D:/vscode项目/Tabular-benchmark/results/results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

# ===================== 统一数据预处理 =====================
rows = []
for dataset_name, data_info in results.items():
    n_samples = data_info["n_samples"]
    task_type = data_info["task_type"]
    for model_name, model_res in data_info["models"].items():
        rows.append({
            "dataset": dataset_name,
            "task_type": task_type,
            "n_samples": n_samples,
            "model": model_name,
            "model_type": model_res["model_type"],
            "score": model_res["score"]
        })
df = pd.DataFrame(rows)
df = df.sort_values("n_samples", ascending=True).reset_index(drop=True)

# ==============================================================================
# Figure 1: 分类 + 回归模型性能对比柱状图
# ==============================================================================
def plot_figure1():
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 7))

    # 分类
    df_cls = df[df["task_type"] == "classification"]
    sns.barplot(data=df_cls, x="dataset", y="score", hue="model", palette=color_map, ax=ax1, edgecolor="white", linewidth=0.6)
    ax1.set_title("Classification Tasks (Sorted by Dataset Size: Small → Large)", fontweight="bold", fontsize=12)
    ax1.set_ylabel("ROC-AUC Score", fontsize=11)
    ax1.set_xlabel("")
    ax1.tick_params(axis="x", rotation=45)
    ax1.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax1.set_ylim(0.5, 1.0)

    # 回归
    df_reg = df[df["task_type"] == "regression"]
    sns.barplot(data=df_reg, x="dataset", y="score", hue="model", palette=color_map, ax=ax2, edgecolor="white", linewidth=0.6)
    ax2.set_title("Regression Tasks (Sorted by Dataset Size: Small → Large)", fontweight="bold", fontsize=12)
    ax2.set_ylabel("R² Score", fontsize=11)
    ax2.set_xlabel("")
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax2.set_ylim(-0.1, 1.0)

    plt.tight_layout()
    plt.savefig("figures/figure1_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Figure1 已保存")

# ==============================================================================
# Figure 2: 模型平均排名
# ==============================================================================
def plot_figure2():
    def get_rank(group):
        group["rank"] = group["score"].rank(ascending=False, method="min")
        return group

    df_rank = df.groupby(["dataset", "task_type"]).apply(get_rank).reset_index(drop=True)
    avg_rank = df_rank.groupby(["model", "model_type"])["rank"].agg(["mean", "std"]).round(2)
    avg_rank = avg_rank.sort_values("mean", ascending=True).reset_index()

    bar_color = [color_map[m] for m in avg_rank["model"]]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(avg_rank["model"], avg_rank["mean"], color=bar_color, edgecolor="white", linewidth=0.8)

    ax.set_ylim(1, 8)
    ax.invert_yaxis()
    ax.set_title("Average Ranking of Models Across All Datasets", fontweight="bold", fontsize=13)
    ax.set_ylabel("Mean Rank (1 = Best, 8 = Worst)", fontsize=11)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)

    for bar, rank_val in zip(bars, avg_rank["mean"]):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.15,
                f"{rank_val}", ha="center", fontsize=9, fontweight="bold")

    ax.axhline(y=4.5, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("figures/figure2_average_ranking.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Figure2 已保存")

# ==============================================================================
# Figure 3: 树模型 vs 深度学习 性能差距散点图
# ==============================================================================
def plot_figure3():
    grouped = df.groupby(["dataset", "task_type", "n_samples", "model_type"])["score"].mean().unstack()
    grouped["gap"] = grouped["tree-based"] - grouped["deep-learning"]
    grouped = grouped.reset_index()

    plt.figure(figsize=(12, 7))
    marker_map = {"classification": "o", "regression": "^"}
    color_map_task = {"classification": "#2E7D32", "regression": "#E64A19"}

    for task in grouped["task_type"].unique():
        sub = grouped[grouped["task_type"] == task]
        plt.scatter(sub["n_samples"], sub["gap"], s=110, marker=marker_map[task],
                    c=color_map_task[task], edgecolor="white", linewidth=1.2, label=task.capitalize())

    plt.xscale("log")
    plt.axhline(y=0, linestyle="--", color="black", alpha=0.7, label="Gap = 0 (Equal Performance)")

    for _, row in grouped.iterrows():
        plt.text(row["n_samples"] * 1.08, row["gap"], row["dataset"], fontsize=8, ha="left")

    plt.title("Tree-vs-Deep Performance Gap vs Dataset Size", fontweight="bold", fontsize=13)
    plt.xlabel("Dataset Size (Log Scale)", fontsize=11)
    plt.ylabel("Average Tree Score − Average Deep Score", fontsize=11)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/figure3_gap_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Figure3 已保存")

# ==============================================================================
# 一键运行所有图
# ==============================================================================
if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)

    print("🚀 开始生成所有图表...")
    plot_figure1()
    plot_figure2()
    plot_figure3()
    print("\n🎉 所有图片生成完成！已保存到 figures/ 文件夹")