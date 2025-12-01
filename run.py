import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def run_training():
    base_dir = Path(__file__).resolve().parent

    dataset_name = "ad"
    train_csv = base_dir / "datasets" / "ad" / "ad_train.csv"
    test_csv  = base_dir / "datasets" / "ad" / "ad_test.csv"
    img_path  = base_dir / "datasets" / "ad" / "all_images"
    save_root = base_dir / "saved_adapter_ad"

    script_path = base_dir / "CMA_fewshot.py"

    shots = [2, 8, 16, 32, 64]
    seeds = range(1, 11)

    RESAMPLE = 0  # 必须保持 0 → 固定 few-shot 子集，保证稳定性

    for p in [train_csv, test_csv, img_path, script_path]:
        if not p.exists():
            raise FileNotFoundError(f"路径不存在: {p}")

    save_root.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    # ⭐ 是否启用 FEAT
    USE_FEAT = True           # False = baseline / True = FEAT
    USE_PROTO_MLP = True      # False = FEAT / True = FEAT + ProtoMLP

    for shot in shots:
        for seed in seeds:
            run_name = f"ad_shot{shot}_seed{seed}"
            save_path = save_root / run_name
            save_path.mkdir(parents=True, exist_ok=True)

            print(f"\n=== Training start: dataset={dataset_name}, shot={shot}, seed={seed} ===\n")

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

            # ========== ⭐ 核心：启用 FEAT 与 ProtoMLP ==========

            if USE_FEAT:
                cmd += ["--use_feat", "--feat_heads", "4", "--feat_layers", "1"]

            if USE_FEAT and USE_PROTO_MLP:
                cmd += ["--proto_mlp"]

            # ====================================================

            # 保存日志
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
