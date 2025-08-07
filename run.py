import os
import subprocess

# 设置训练参数
shots = [2, 8, 16, 32]  # 训练的 shot 数量
seeds = range(1, 11)  # 训练的随机种子

# 定义路径和保存文件夹
train_csv = "./datasets/weibo/weibo_train.csv"
test_csv = "./datasets/weibo/weibo_test.csv"
img_path = "./datasets/weibo/all_images/"
save_base_path = "./saved_adapter/"

# 遍历每个 shot 和 seed
for shot in shots:
    for seed in seeds:
        save_path = os.path.join(save_base_path, f"weibo_shot{shot}_seed{seed}")
        os.makedirs(save_path, exist_ok=True)

        # 训练命令
        cmd = [
            "python", "CMA_fewshot.py",
            "--seed", str(seed),
            "--dataset_name", "weibo",
            "--train_csv", train_csv,
            "--test_csv", test_csv,
            "--img_path", img_path,
            "--shot", str(shot),
            "--save_path", save_path
        ]

        print(f"\n=== Running Experiment: shot={shot}, seed={seed} ===")

        # 执行命令并显示输出
        result = subprocess.run(cmd, capture_output=True, text=True)

        # 打印训练过程和结果
        print(f"Training Output for shot={shot}, seed={seed}:\n")
        print(result.stdout)  # 打印训练过程输出
        print(result.stderr)  # 打印可能的错误信息

        # 如果训练过程中产生了日志或结果文件，可以读取并打印
        log_file = os.path.join(save_path, "log.txt")  # 假设训练日志会存储为 log.txt
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as log:
                print(f"Log file for shot={shot}, seed={seed}:\n")
                print(log.read())

        # 打印训练完成后相关的模型性能指标
        # 假设训练脚本会保存模型评估结果，如 `results.json` 或 `accuracy.txt`
        accuracy_file = os.path.join(save_path, "accuracy.txt")  # 假设准确率保存在 accuracy.txt
        if os.path.exists(accuracy_file):
            with open(accuracy_file, "r", encoding="utf-8") as acc_file:
                accuracy = acc_file.read()
                print(f"Test Accuracy for shot={shot}, seed={seed}: {accuracy}\n")
        else:
            print(f"No accuracy file found for shot={shot}, seed={seed}\n")
