import os
import subprocess


TRAIN_CSV = "/home/zhaopanzhi/CMA/FND_fewshot-main/datasets/ad/ad_train.csv"
TEST_CSV = "/home/zhaopanzhi/CMA/FND_fewshot-main/datasets/ad/ad_test.csv"
IMG_PATH = "/home/zhaopanzhi/CMA/FND_fewshot-main/datasets/ad/all_images/"

SAVE_PATH = "./checkpoints/ad_proto_best_structure_v1"
EXP_TAG = "ad_proto_best_structure_v1"

EPOCHS = 20
BATCH_SIZE = 8
TEST_BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-2
NUM_PROTOTYPES = 2
PROTO_TEMPERATURE = 1.0
PROTO_CE_WEIGHT = 0.0
PROTO_ALIGN_WEIGHT = 0.0
PROTO_DIV_WEIGHT = 0.0

# Existing key ablations already finished: ["full", "no_proto", "no_conflict_bias"].
# Run the remaining paper-table ablations by default.
# Full list: ["full", "no_proto", "no_conflict_bias", "text_only_proto", "no_sadg", "mean_pool"]
ABLATIONS = ["text_only_proto", "no_sadg", "mean_pool"]

SHOTS = [2, 8, 16, 32]
SEEDS = range(1, 11)


def build_command(shot, seed, ablation, save_path, exp_tag):
    return [
        "python", "/home/zhaopanzhi/CMA/FND_fewshot-main/CMA_fewshot.py",
        "--dataset_name", "ad",
        "--train_csv", TRAIN_CSV,
        "--test_csv", TEST_CSV,
        "--img_path", IMG_PATH,
        "--seed", str(seed),
        "--shot", str(shot),
        "--save_path", save_path,
        "--exp_tag", exp_tag,
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH_SIZE),
        "--test_batch_size", str(TEST_BATCH_SIZE),
        "--lr", str(LR),
        "--weight_decay", str(WEIGHT_DECAY),
        "--num_prototypes", str(NUM_PROTOTYPES),
        "--proto_temperature", str(PROTO_TEMPERATURE),
        "--proto_ce_weight", str(PROTO_CE_WEIGHT),
        "--proto_align_weight", str(PROTO_ALIGN_WEIGHT),
        "--proto_div_weight", str(PROTO_DIV_WEIGHT),
        "--ablation", ablation,
    ]


def run_experiment():
    os.makedirs(SAVE_PATH, exist_ok=True)

    for ablation in ABLATIONS:
        os.makedirs(SAVE_PATH, exist_ok=True)

        for shot in SHOTS:
            for seed in SEEDS:
                print("\n" + "=" * 50)
                print(
                    f"Running Experiment: Ablation={ablation}, "
                    f"Shot={shot}, Seed={seed}, Tag={EXP_TAG}"
                )
                print("=" * 50 + "\n")

                cmd = build_command(
                    shot=shot,
                    seed=seed,
                    ablation=ablation,
                    save_path=SAVE_PATH,
                    exp_tag=EXP_TAG,
                )

                try:
                    print("Command:", " ".join(cmd))
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error at Ablation={ablation}, Shot={shot}, Seed={seed}")
                    print(e)
                    continue


if __name__ == "__main__":
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(TEST_CSV):
        print(f"Missing csv file: {TRAIN_CSV} or {TEST_CSV}")
    elif not os.path.exists(IMG_PATH):
        print(f"Missing image directory: {IMG_PATH}")
    else:
        run_experiment()
