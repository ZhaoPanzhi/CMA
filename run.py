import subprocess
import os

# ================= 配置区域 =================
# 1. 数据集路径 (请确保指向您生成的那个平衡数据集!)
TRAIN_CSV = "E:\\CMA\\FND_fewshot-main\\datasets\\ad\\ad_train.csv"
TEST_CSV = "E:\\CMA\\FND_fewshot-main\\datasets\\ad\\ad_test.csv"

# 2. 图片路径 (请修改为您实际存放图片的文件夹路径)
IMG_PATH = "E:\\CMA\\FND_fewshot-main\\datasets\\ad\\all_images/"

# 3. 结果保存路径
SAVE_PATH = "./saved_ad_SliceAttentionFusion_SADG"

# 4. 实验参数
SHOTS = [8, 16, 32]  # 少样本设置
SEEDS = range(1, 11)  # 跑 5 个种子取平均 (1, 2, 3, 4, 5)


# ===========================================

def run_experiment():
    # 确保保存目录存在
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    for shot in SHOTS:
        for seed in SEEDS:
            print(f"\n{'=' * 40}")
            print(f"🚀 Running Baseline: Shot={shot}, Seed={seed}")
            print(f"{'=' * 40}\n")

            cmd = [
                "python", "E:\\CMA\\FND_fewshot-main\\CMA_fewshot.py",
                "--dataset_name", "ad",  # 这里对应 CMA_fewshot.py 里新增的 elif
                "--train_csv", TRAIN_CSV,
                "--test_csv", TEST_CSV,
                "--img_path", IMG_PATH,
                "--seed", str(seed),
                "--shot", str(shot),
                "--save_path", SAVE_PATH,

                # === Baseline 关键参数 ===
                # 既然是 Baseline，通常不需要太大的 Patience，20 足够
                # 也不需要特殊的 Loss 权重，因为数据已经平衡了
            ]

            try:
                # 打印命令方便调试
                print("Command:", " ".join(cmd))

                # 运行命令，check=True 会在脚本出错时抛出异常
                subprocess.run(cmd, check=True)

            except subprocess.CalledProcessError as e:
                print(f"❌ Error occurred at Shot {shot}, Seed {seed}!")
                print(e)
                # 可以选择 continue 继续跑下一个，或者 break 停止
                # continue


if __name__ == "__main__":
    # 检查数据文件是否存在
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(TEST_CSV):
        print(f"❌ 错误: 找不到数据文件！请确认 {TRAIN_CSV} 和 {TEST_CSV} 在当前目录下。")
    else:
        run_experiment()