import os
import subprocess

def run_training():
    dataset_name = "weibo"
    train_csv = "./datasets/weibo/weibo_train.csv"
    test_csv = "./datasets/weibo/weibo_test.csv"
    img_path = "./datasets/weibo/all_images/"
    save_root = "./saved_adapter"

    shots = [2, 8, 16, 32]
    seeds = range(1, 11)  # 1 到 10

    for shot in shots:
        for seed in seeds:
            save_path = os.path.join(save_root, f"weibo_shot{shot}_seed{seed}")
            print(f"\n=== Training start: dataset={dataset_name}, shot={shot}, seed={seed} ===\n")

            cmd = [
                "python", script_path,
                "--seed", str(seed),
                "--dataset_name", dataset_name,
                "--train_csv", train_csv,
                "--test_csv", test_csv,
                "--img_path", img_path,
                "--shot", str(shot),
                "--save_path", save_path
            ]

            # 调用训练命令
            subprocess.run(cmd)


if __name__ == "__main__":
    run_training()
