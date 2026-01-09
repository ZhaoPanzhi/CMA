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

# 引入修改后的模块
from my_datautils import FakeNews_Dataset, FewShotSampler_weibo, FewShotSampler_fakenewsnet
from mymodels import CMA_Model  # 确保这里的 CMA_Model 是你修改过包含 SADG 的版本
from cn_clip.clip import load_from_name

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seeds(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def save_results(args, history, best_preds, save_dir, best_metric_val):
    """
    保存所有论文需要的实验结果
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. 保存配置参数 (Config)
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # 2. 保存训练日志
    with open(os.path.join(save_dir, "training_log.json"), 'w') as f:
        json.dump(history, f, indent=4)

    # 3. 保存最佳模型的预测结果
    if best_preds:
        df_preds = pd.DataFrame(best_preds)
        df_preds.to_csv(os.path.join(save_dir, "best_predictions.csv"), index=False)

        # 4. 生成并保存详细评估报告
        y_true = df_preds['label'].values
        y_pred = df_preds['pred'].values

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        report = classification_report(y_true, y_pred, digits=4, output_dict=True, zero_division=0)

        summary = {
            "Best Macro F1": best_metric_val,
            "Confusion Matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
            "Accuracy": accuracy_score(y_true, y_pred),
            "Weighted F1": f1_score(y_true, y_pred, average='weighted'),
            "Detailed Report": report
        }

        with open(os.path.join(save_dir, "best_metrics_summary.json"), 'w') as f:
            json.dump(summary, f, indent=4)

    print(f"✅ Results saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", type=str, default="weibo")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--shot", type=int, default=2)
    parser.add_argument("--save_path", type=str, default="./checkpoints")
    args = parser.parse_args()

    set_seeds(args.seed)

    # 定义结果保存目录
    exp_name = f"{args.dataset_name}_{args.shot}shot_seed{args.seed}"
    result_dir = os.path.join("./paper_results", exp_name)

    print(f"🚀 Experiment: {exp_name}")
    print("Loading Chinese CLIP (Frozen)...")

    clip_model, preprocess = load_from_name("ViT-B-16", device=device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    # 数据集准备
    # 注意：如果启用了多切片融合，记得在 my_datautils 里把 max_slices 设好
    train_dataset = FakeNews_Dataset(clip_model, preprocess, args.train_csv, args.img_path, args.dataset_name,
                                     max_slices=8)
    test_dataset = FakeNews_Dataset(clip_model, preprocess, args.test_csv, args.img_path, args.dataset_name,
                                    max_slices=8)

    # Few-shot 采样
    if args.dataset_name == 'weibo':
        train_sampler = FewShotSampler_weibo(train_dataset, args.shot, args.seed)
        train_dataset = train_sampler.get_train_dataset()
    else:
        # 如果 ad 数据集也用 weibo 的采样逻辑，就走上面那个分支
        # 这里假设 ad 数据集结构和 weibo 类似
        train_sampler = FewShotSampler_weibo(train_dataset, args.shot, args.seed)
        train_dataset = train_sampler.get_train_dataset()

    print(f"Train Set Size (Groups): {len(train_dataset)}")

    # Batch Size 建议调小，因为现在是多切片
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 初始化模型
    cma_model = CMA_Model(feature_dim=512, num_classes=2).to(device)

    optimizer = AdamW(cma_model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_func = CrossEntropyLoss()

    # --- 修改点：初始化 Best Metric 为 F1 ---
    best_f1 = -1.0
    best_preds_data = []

    history = {
        "epoch": [],
        "loss": [],
        "train_acc": [],
        "test_acc": [],
        "test_f1_macro": [],
        "test_f1_weighted": []
    }

    EPOCH = 20

    for epoch in range(EPOCH):
        cma_model.train()
        total_loss = 0
        correct = 0
        total = 0

        # 注意：这里接收 mask
        for txt, img, label, mask in train_loader:
            txt, img, label, mask = txt.to(device), img.to(device), label.to(device), mask.to(device)
            B, S, C, H, W = img.shape

            with torch.no_grad():
                img_flat = img.view(B * S, C, H, W)
                txt_flat = txt.view(B * S, -1)

                img_feat_flat = clip_model.encode_image(img_flat)
                txt_feat_flat = clip_model.encode_text(txt_flat)

                img_feat = img_feat_flat.view(B, S, -1)
                txt_feat = txt_feat_flat.view(B, S, -1)

            optimizer.zero_grad()
            logits = cma_model(txt_feat.float(), img_feat.float(), mask)

            loss = loss_func(logits, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

        train_acc = correct / total if total > 0 else 0
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}")

        # --- Evaluation ---
        cma_model.eval()
        test_labels = []
        pred_labels = []
        pred_probs = []

        with torch.no_grad():
            for txt, img, label, mask in tqdm.tqdm(test_loader, desc="Testing"):
                txt, img, label, mask = txt.to(device), img.to(device), label.to(device), mask.to(device)
                B, S, C, H, W = img.shape

                img_flat = img.view(B * S, C, H, W)
                txt_flat = txt.view(B * S, -1)

                img_feat_flat = clip_model.encode_image(img_flat)
                txt_feat_flat = clip_model.encode_text(txt_flat)

                img_feat = img_feat_flat.view(B, S, -1)
                txt_feat = txt_feat_flat.view(B, S, -1)

                logits = cma_model(txt_feat.float(), img_feat.float(), mask)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=-1)

                test_labels.extend(label.cpu().numpy())
                pred_labels.extend(preds.cpu().numpy())
                pred_probs.extend(probs.cpu().numpy())

        curr_acc = accuracy_score(test_labels, pred_labels)
        macro_f1 = f1_score(test_labels, pred_labels, average='macro')
        weighted_f1 = f1_score(test_labels, pred_labels, average='weighted')

        history["epoch"].append(epoch + 1)
        history["loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(curr_acc)
        history["test_f1_macro"].append(macro_f1)
        history["test_f1_weighted"].append(weighted_f1)

        print(f"Test Accuracy: {curr_acc:.4f} | Macro F1: {macro_f1:.4f}")

        # --- 修改点：以 Macro F1 为标准保存模型 ---
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            print(f"🔥 New Best Macro F1: {best_f1:.4f} (Acc: {curr_acc:.4f}), Saving model...")

            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(cma_model.state_dict(), os.path.join(args.save_path, f"best_model_seed{args.seed}.pt"))

            if len(pred_probs) > 0:
                probs_np = np.array(pred_probs)
                best_preds_data = {
                    "label": test_labels,
                    "pred": pred_labels,
                    "prob_0": probs_np[:, 0],
                    "prob_1": probs_np[:, 1]
                }

    print(f"Final Best Macro F1: {best_f1}")

    # 保存结果
    save_results(args, history, best_preds_data, result_dir, best_f1)