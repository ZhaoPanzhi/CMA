import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def run_training():
    # 以当前文件为基准，自动拼路径（跨平台）
    base_dir = Path(__file__).resolve().parent

    dataset_name = "weibo"
    train_csv = base_dir / "datasets" / "weibo" / "weibo_train.csv"
    test_csv  = base_dir / "datasets" / "weibo" / "weibo_test.csv"
    img_path  = base_dir / "datasets" / "weibo" / "all_images"
    save_root = base_dir / "saved_adapter"

    # 通用写法（如果脚本就在本项目目录）
    script_path = base_dir / "CMA_fewshot.py"

    shots = [2, 8, 16, 32]
    seeds = range(1, 11)  # 1 到 10

    RESAMPLE = 1  # ✅ 是否每次随机采样：0=固定可复现，1=开启随机模式

    # 检查关键路径
    for p in [train_csv, test_csv, img_path, script_path]:
        if not p.exists():
            raise FileNotFoundError(f"路径不存在: {p}")

    save_root.mkdir(parents=True, exist_ok=True)

    # 使用当前 Python 解释器，避免系统上有多个 python 导致环境不一致
    py = sys.executable

    for shot in shots:
        for seed in seeds:
            run_name = f"weibo_shot{shot}_seed{seed}"
            save_path = save_root / run_name
            save_path.mkdir(parents=True, exist_ok=True)

            print(f"\n=== Training start: dataset={dataset_name}, shot={shot}, seed={seed} ===\n")

            cmd = [
                py, str(script_path),
                "--seed", str(seed),
                "--dataset_name", dataset_name,
                "--train_csv", str(train_csv),
                "--test_csv", str(test_csv),
                "--img_path", str(img_path) + os.sep,  # 末尾带分隔符更稳
                "--shot", str(shot),
                "--save_path", str(save_path),
                # 如果你集成了 FEAT，打开下面两行：
                "--use_feat",
                "--feat_heads", "4", "--feat_layers", "1",
                # 随机采样
                "--resample", str(RESAMPLE),
            ]

            # 将 stdout/stderr 保存到日志，便于排错与汇总
            log_file = save_path / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

            with open(log_file, "w", encoding="utf-8") as lf:
                lf.write(" ".join(cmd) + "\n\n")
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",  # ✅ 关键行
                    errors="ignore"  # ✅ 防止极个别符号再报错
                )
                for line in proc.stdout:
                    print(line, end="")  # 实时输出
                proc.wait()

            if proc.returncode != 0:
                # 自动把日志尾部打印出来
                print(f"[WARN] 运行失败（返回码 {proc.returncode}）：{run_name}，日志：{log_file}")
                try:
                    tail = Path(log_file).read_text(encoding="utf-8").splitlines()[-120:]
                    print("\n----- LOG TAIL -----")
                    print("\n".join(tail))
                    print("----- END LOG TAIL -----\n")
                except Exception as e:
                    print(f"[WARN] 读取日志失败: {e}")
            else:
                print(f"[OK] 完成：{run_name}，日志：{log_file}")

if __name__ == "__main__":
    run_training()
