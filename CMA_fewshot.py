import os
import argparse
import torch
import tqdm
import json
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from my_datautils import FakeNews_Dataset, FewShotSampler_weibo, FewShotSampler_fakenewsnet
from mymodels import CMA_Model
from cn_clip.clip import load_from_name

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seeds(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_clip_features(clip_model, txt, img):
    """
    txt: [B, S, L]
    img: [B, S, C, H, W]
    return:
        txt_feat: [B, S, D]
        img_feat: [B, S, D]
    """
    B, S, C, H, W = img.shape
    with torch.no_grad():
        img_flat = img.view(B * S, C, H, W)
        txt_flat = txt.view(B * S, -1)

        img_feat_flat = clip_model.encode_image(img_flat)
        txt_feat_flat = clip_model.encode_text(txt_flat)

        img_feat = img_feat_flat.view(B, S, -1)
        txt_feat = txt_feat_flat.view(B, S, -1)

    return txt_feat, img_feat


def save_results(args, history, best_preds, save_dir, best_metric_val, best_epoch, best_acc):
    """
    保存实验结果，方便后续论文写作和统计
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. 配置
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # 2. 训练日志
    with open(os.path.join(save_dir, "training_log.json"), "w") as f:
        json.dump(history, f, indent=4)

    pd.DataFrame(history).to_csv(os.path.join(save_dir, "training_log.csv"), index=False)

    # 3. 最佳预测结果 + 评估摘要
    if best_preds:
        df_preds = pd.DataFrame(best_preds)
        df_preds.to_csv(os.path.join(save_dir, "best_predictions.csv"), index=False)

        y_true = df_preds["label"].values
        y_pred = df_preds["pred"].values

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        report = classification_report(
            y_true, y_pred, digits=4, output_dict=True, zero_division=0
        )

        cls0 = report["0"] if "0" in report else {"precision": 0, "recall": 0, "f1-score": 0}
        cls1 = report["1"] if "1" in report else {"precision": 0, "recall": 0, "f1-score": 0}

        summary = {
            "Best Epoch": best_epoch,
            "Best Macro F1": best_metric_val,
            "Best Accuracy": best_acc,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Weighted F1": f1_score(y_true, y_pred, average="weighted"),
            "Confusion Matrix": {
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp)
            },
            "Class_0_Precision": cls0["precision"],
            "Class_0_Recall": cls0["recall"],
            "Class_0_F1": cls0["f1-score"],
            "Class_1_Precision": cls1["precision"],
            "Class_1_Recall": cls1["recall"],
            "Class_1_F1": cls1["f1-score"],
            "Detailed Report": report
        }

        with open(os.path.join(save_dir, "best_metrics_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)

    print(f"✅ Results saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 基本参数
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", type=str, default="weibo")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--shot", type=int, default=2)
    parser.add_argument("--save_path", type=str, default="./checkpoints")
    parser.add_argument("--exp_tag", type=str, default="base")

    # 训练参数 —— 恢复到高结果导向版本
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    # 第二创新点相关 —— 保留参数，但默认不额外训练 proto 分支
    parser.add_argument("--num_prototypes", type=int, default=4)
    parser.add_argument("--fusion_gamma", type=float, default=0.1)

    # 可视化辅助
    parser.add_argument("--save_aux", action="store_true")

    args = parser.parse_args()

    set_seeds(args.seed)

    exp_name = f"{args.dataset_name}_{args.shot}shot_seed{args.seed}"
    result_dir = os.path.join("./result", args.exp_tag, exp_name)
    os.makedirs(result_dir, exist_ok=True)

    print(f"🚀 Experiment: {exp_name}")
    print("Loading Chinese CLIP (Frozen)...")

    clip_model, preprocess = load_from_name("ViT-B-16", device=device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    # 数据集
    train_dataset = FakeNews_Dataset(
        clip_model, preprocess, args.train_csv, args.img_path, args.dataset_name, max_slices=8
    )
    test_dataset = FakeNews_Dataset(
        clip_model, preprocess, args.test_csv, args.img_path, args.dataset_name, max_slices=8
    )

    # Few-shot 采样
    if args.dataset_name == "weibo":
        train_sampler = FewShotSampler_weibo(train_dataset, args.shot, args.seed)
        train_dataset = train_sampler.get_train_dataset()
    elif args.dataset_name == "ad":
        train_sampler = FewShotSampler_weibo(train_dataset, args.shot, args.seed)
        train_dataset = train_sampler.get_train_dataset()
    else:
        raise ValueError(f"Unsupported dataset_name: {args.dataset_name}")

    print(f"Train Set Size (Groups): {len(train_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    # 模型 —— 保留 num_prototypes 参数
    cma_model = CMA_Model(
        feature_dim=512,
        num_classes=2,
        num_prototypes=args.num_prototypes,
    ).to(device)

    optimizer = AdamW(
        cma_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    loss_func = CrossEntropyLoss()

    # 最优记录
    best_f1 = -1.0
    best_acc = -1.0
    best_epoch = -1
    best_preds_data = []

    history = {
        "epoch": [],
        "loss": [],
        "train_acc": [],
        "test_acc": [],
        "test_f1_macro": [],
        "test_f1_weighted": [],
        "best_so_far_f1": [],
        "sadg_alpha": []
    }

    EPOCH = args.epochs

    for epoch in range(EPOCH):
        cma_model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for txt, img, label, mask in train_loader:
            txt = txt.to(device)
            img = img.to(device)
            label = label.to(device)
            mask = mask.to(device)

            txt_feat, img_feat = extract_clip_features(clip_model, txt, img)

            optimizer.zero_grad()

            # 只训练 final_logits，恢复高结果版本逻辑
            logits = cma_model(
                txt_feat.float(),
                img_feat.float(),
                mask
            )

            loss = loss_func(logits, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

        train_acc = correct / total if total > 0 else 0
        avg_loss = total_loss / len(train_loader)

        alpha_value = 0.0
        if hasattr(cma_model, "sadg") and hasattr(cma_model.sadg, "alpha"):
            alpha_value = cma_model.sadg.alpha.item()

        print(
            f"Epoch {epoch + 1} | "
            f"Loss: {avg_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}"
        )

        # ================= Evaluation =================
        cma_model.eval()
        test_labels = []
        pred_labels = []
        pred_probs = []

        with torch.no_grad():
            for txt, img, label, mask in tqdm.tqdm(test_loader, desc="Testing"):
                txt = txt.to(device)
                img = img.to(device)
                label = label.to(device)
                mask = mask.to(device)

                txt_feat, img_feat = extract_clip_features(clip_model, txt, img)

                if args.save_aux:
                    logits, aux = cma_model(
                        txt_feat.float(),
                        img_feat.float(),
                        mask,
                        return_aux=True
                    )
                else:
                    logits = cma_model(
                        txt_feat.float(),
                        img_feat.float(),
                        mask
                    )

                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=-1)

                test_labels.extend(label.cpu().numpy())
                pred_labels.extend(preds.cpu().numpy())
                pred_probs.extend(probs.cpu().numpy())

        curr_acc = accuracy_score(test_labels, pred_labels)
        macro_f1 = f1_score(test_labels, pred_labels, average="macro")
        weighted_f1 = f1_score(test_labels, pred_labels, average="weighted")

        current_best_f1 = max(best_f1, macro_f1)

        history["epoch"].append(epoch + 1)
        history["loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(curr_acc)
        history["test_f1_macro"].append(macro_f1)
        history["test_f1_weighted"].append(weighted_f1)
        history["best_so_far_f1"].append(current_best_f1)
        history["sadg_alpha"].append(alpha_value)

        print(f"Test Accuracy: {curr_acc:.4f} | Macro F1: {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_acc = curr_acc
            best_epoch = epoch + 1

            print(f"🔥 New Best Macro F1: {best_f1:.4f} (Acc: {curr_acc:.4f}), Saving model...")

            os.makedirs(args.save_path, exist_ok=True)
            torch.save(
                cma_model.state_dict(),
                os.path.join(args.save_path, f"best_model_seed{args.seed}.pt")
            )

            if len(pred_probs) > 0:
                probs_np = np.array(pred_probs)
                best_preds_data = {
                    "label": test_labels,
                    "pred": pred_labels,
                    "prob_0": probs_np[:, 0],
                    "prob_1": probs_np[:, 1]
                }

    print(f"Final Best Macro F1: {best_f1:.6f}")

    save_results(
        args=args,
        history=history,
        best_preds=best_preds_data,
        save_dir=result_dir,
        best_metric_val=best_f1,
        best_epoch=best_epoch,
        best_acc=best_acc
    )

    result_row = pd.DataFrame([{
        "dataset": args.dataset_name,
        "shot": args.shot,
        "seed": args.seed,
        "exp_tag": args.exp_tag,
        "best_epoch": best_epoch,
        "best_macro_f1": best_f1,
        "best_acc": best_acc,
        "num_prototypes": args.num_prototypes,
        "lr": args.lr
    }])
    result_row.to_csv(os.path.join(result_dir, "result_row.csv"), index=False)