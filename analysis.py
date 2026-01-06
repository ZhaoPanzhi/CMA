import os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===== 配置路径 =====
SAVE_ROOT = "/home/zhaopanzhi/CMA/FND_fewshot-main/saved_cma_ACFC_ad01"
PATTERN = "ad_shot*_seed*"


# ===== 工具函数 =====
def load_json(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None


def load_csv(p):
    try:
        return pd.read_csv(p)
    except:
        return None


# ===== 扫描所有 run =====
run_dirs = sorted(glob.glob(os.path.join(SAVE_ROOT, PATTERN)))
rows = []

for rd in run_dirs:
    name = os.path.basename(rd)
    try:
        shot = int(name.split("shot")[1].split("_")[0])
        seed = int(name.split("seed")[1])
    except:
        continue

    best = load_json(os.path.join(rd, "best_metrics.json"))
    metrics = load_csv(os.path.join(rd, "metrics_epoch.csv"))

    if best:
        best_acc = best.get("accuracy", np.nan)
        best_epoch = best.get("epoch", np.nan)
        best_f1 = best.get("macro_avg", {}).get("f1-score", np.nan)
    else:
        best_acc, best_epoch, best_f1 = np.nan, np.nan, np.nan

    last_acc = last_f1 = np.nan
    if metrics is not None and len(metrics) > 0:
        last_acc = metrics["accuracy"].iloc[-1]
        last_f1 = metrics["macro_f1"].iloc[-1]

    rows.append({
        "run_dir": name,
        "shot": shot,
        "seed": seed,
        "best_acc": best_acc,
        "best_macro_f1": best_f1,
        "best_epoch": best_epoch,
        "last_acc": last_acc,
        "last_macro_f1": last_f1,
    })

# ===== 写出总表 =====
summary_df = pd.DataFrame(rows).sort_values(["shot", "seed"]).reset_index(drop=True)
summary_df.to_csv(os.path.join(SAVE_ROOT, "summary_runs.csv"), index=False)
print("[OK] summary_runs.csv 已生成")


# ===== 按 shot 聚合 =====
def stats(x):
    return pd.Series({"mean": x.mean(), "std": x.std(ddof=1)})


agg = []
for shot, g in summary_df.groupby("shot"):
    acc_stats = stats(g["best_acc"])
    f1_stats = stats(g["best_macro_f1"])

    best_item = g.sort_values(["best_acc", "best_macro_f1"], ascending=False).iloc[0]
    agg.append({
        "shot": shot,
        "n_seeds": len(g),
        "acc_mean": acc_stats["mean"],
        "acc_std": acc_stats["std"],
        "macro_f1_mean": f1_stats["mean"],
        "macro_f1_std": f1_stats["std"],
        "best_seed": int(best_item["seed"]),
        "best_acc": best_item["best_acc"],
        "best_macro_f1": best_item["best_macro_f1"],
    })

by_shot = pd.DataFrame(agg).sort_values("shot")
by_shot.to_csv(os.path.join(SAVE_ROOT, "summary_by_shot.csv"), index=False)
print("[OK] summary_by_shot.csv 已生成")

# ===== 可视化：箱线图 =====
shots = sorted(summary_df["shot"].unique())
plt.figure(figsize=(8, 4))
plt.boxplot([summary_df[summary_df.shot == s]["best_acc"] for s in shots], labels=shots)
plt.title("Best Accuracy by Shot");
plt.xlabel("Shot");
plt.ylabel("Accuracy")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig(os.path.join(SAVE_ROOT, "acc_box_by_shot.png"));
plt.close()

plt.figure(figsize=(8, 4))
plt.boxplot([summary_df[summary_df.shot == s]["best_macro_f1"] for s in shots], labels=shots)
plt.title("Best Macro-F1 by Shot");
plt.xlabel("Shot");
plt.ylabel("Macro-F1")
plt.grid(True, linestyle="--", alpha=0.5)
plt.savefig(os.path.join(SAVE_ROOT, "f1_box_by_shot.png"));
plt.close()

# ===== 均值折线图（适合论文） =====
plt.figure(figsize=(7, 4))
plt.plot(by_shot["shot"], by_shot["acc_mean"], marker="o", label="Acc Mean")
plt.fill_between(by_shot["shot"],
                 by_shot["acc_mean"] - by_shot["acc_std"],
                 by_shot["acc_mean"] + by_shot["acc_std"],
                 alpha=0.2)
plt.title("Accuracy Mean ± Std by Shot");
plt.xlabel("Shot");
plt.ylabel("Accuracy")
plt.grid(True);
plt.legend()
plt.savefig(os.path.join(SAVE_ROOT, "acc_line_by_shot.png"));
plt.close()

plt.figure(figsize=(7, 4))
plt.plot(by_shot["shot"], by_shot["macro_f1_mean"], marker="s", label="Macro-F1 Mean", color="orange")
plt.fill_between(by_shot["shot"],
                 by_shot["macro_f1_mean"] - by_shot["macro_f1_std"],
                 by_shot["macro_f1_mean"] + by_shot["macro_f1_std"],
                 color="orange", alpha=0.2)
plt.title("Macro-F1 Mean ± Std by Shot");
plt.xlabel("Shot");
plt.ylabel("Macro-F1")
plt.grid(True);
plt.legend()
plt.savefig(os.path.join(SAVE_ROOT, "f1_line_by_shot.png"));
plt.close()

# ===== 打印最终分析报告 =====
print("\n=== 按 Shot 聚合统计 ===")
print(by_shot.to_string(index=False))

print("\n=== 异常 Seed 检测（低于均值 2*std 的视为异常） ===")
for _, row in by_shot.iterrows():
    shot = row["shot"]
    mean, std = row["acc_mean"], row["acc_std"]
    low_threshold = mean - 2 * std
    bad = summary_df[(summary_df.shot == shot) & (summary_df.best_acc < low_threshold)]
    if len(bad) > 0:
        print(f"\nShot {shot}: 发现异常 seeds：")
        print(bad[["seed", "best_acc"]].to_string(index=False))
    else:
        print(f"Shot {shot}: 无异常 seed")

# ================================================================
# ========== 方案 C：生成论文风格的 2×2 子图（你提供的风格） ==========
# ================================================================

print("\n=== 正在生成 2×2 子图（论文风格） ===")

paper_out_dir = os.path.join(SAVE_ROOT, "paper_style_plots")
os.makedirs(paper_out_dir, exist_ok=True)

# 统一获取所有 shots
shots_sorted = sorted(summary_df["shot"].unique())

# 如果 shots 个数不是 4，也自动适应
num_shots = len(shots_sorted)
cols = 2
rows = (num_shots + 1) // 2

fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
axes = axes.flatten()

for idx, shot in enumerate(shots_sorted):
    ax = axes[idx]
    group = summary_df[summary_df.shot == shot]

    # 默认只取 seed 最小的 run（你图中的每个 shot 对应一个 seed）
    row = group.sort_values("seed").iloc[0]
    run_dir = os.path.join(SAVE_ROOT, row["run_dir"])
    curve_csv = os.path.join(run_dir, "metrics_epoch.csv")
    curve = load_csv(curve_csv)

    if curve is None or len(curve) == 0:
        continue

    seed = int(row["seed"])
    epochs = curve["epoch"]
    acc = curve["accuracy"]
    f1 = curve["macro_f1"]

    # 绘制 Accuracy / F1
    ax.plot(epochs, acc, marker="o", label="Accuracy")
    ax.plot(epochs, f1, marker="s", label="Macro-F1")

    ax.set_title(f"Shot={shot} (seed={seed})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

# 如果子图比 shot 多，隐藏空图
for j in range(idx + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
out_png = os.path.join(paper_out_dir, "shots_2x2_overview.png")
plt.savefig(out_png, dpi=200)
plt.close()

print(f"[OK] 已生成 papers 风格图：{out_png}")
