import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def run_training():
    base_dir = Path(__file__).resolve().parent

    # ====== 数据路径 ======
    dataset_name = "ad"
    train_csv = base_dir / "datasets" / "ad" / "ad_train.csv"
    test_csv  = base_dir / "datasets" / "ad" / "ad_test.csv"
    img_path  = base_dir / "datasets" / "ad" / "all_images"
    script_path = base_dir / "CMA_fewshot.py"

    # ====== Few-shot 参数 ======
    shots = [2, 8, 16, 32]
    seeds = range(1, 11)
    RESAMPLE = 0    # 固定 few-shot → 必须为 0

    # ====== 实验模式（你论文所需的所有模型） ======
    MODES = [
        "cma",          # 原 CMA baseline
        # "text_only",    # 文本 baseline
        # "img_only",     # 图像 baseline
        # "mlp_only"      # 不用 FEAT、Adapter，仅用 MLP 的双模态 baseline
    ]

    # ====== 路径检查 ======
    for p in [train_csv, test_csv, img_path, script_path]:
        if not p.exists():
            raise FileNotFoundError(f"路径不存在: {p}")

    py = sys.executable

    # ===========================================================
    #                遍历 6 种模式 × 5 种 shot × 10 个种子
    # ===========================================================
    for mode in MODES:

        # 为每种模式建立单独文件夹
        save_root = base_dir / f"saved_{mode}_CMG_ad_old02"
        save_root.mkdir(parents=True, exist_ok=True)

        print(f"\n============================")
        print(f"开始执行模式：{mode}")
        print(f"============================\n")

        for shot in shots:
            for seed in seeds:

                run_name = f"ad_shot{shot}_seed{seed}"
                save_path = save_root / run_name
                save_path.mkdir(parents=True, exist_ok=True)

                print(f"\n=== Training mode={mode}, shot={shot}, seed={seed} ===\n")

                # ====== 基本参数 ======
                cmd = [
                    py, str(script_path),
                    "--seed", str(seed),
                    "--dataset_name", dataset_name,
                    "--train_csv", str(train_csv),
                    "--test_csv", str(test_csv),
                    "--img_path", str(img_path) + os.sep,
                    "--shot", str(shot),
                    "--save_path", str(save_path),
                    "--resample", str(RESAMPLE),
                ]

                # ====== 模式分支（关键） ======
                if mode == "cma":
                    cmd += ["--mode", "cma"]

                elif mode == "cma_feat":
                    cmd += ["--mode", "cma", "--use_feat"]

                elif mode == "cma_feat_mlp":
                    cmd += ["--mode", "cma", "--use_feat", "--proto_mlp"]

                elif mode == "text_only":
                    cmd += ["--mode", "text_only"]

                elif mode == "img_only":
                    cmd += ["--mode", "img_only"]

                elif mode == "mlp_only":
                    cmd += ["--mode", "mlp_only"]

                # ====== 记录日志文件 ======
                log_file = save_path / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

                with open(log_file, "w", encoding="utf-8") as lf:
                    lf.write(" ".join(cmd) + "\n\n")
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="ignore"
                    )
                    for line in proc.stdout:
                        print(line, end="")
                    proc.wait()

                if proc.returncode != 0:
                    print(f"[WARN] 运行失败：{run_name}")
                else:
                    print(f"[OK] 完成：{run_name}，日志：{log_file}")


if __name__ == "__main__":
    run_training()
