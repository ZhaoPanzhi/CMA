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

# 引入修改后的模块 (确保 mymodels 里有你最新的模型定义)
from my_datautils import FakeNews_Dataset, FewShotSampler_weibo, FewShotSampler_fakenewsnet
from mymodels import CMA_Model  # 或者 CMA_Model_With_ACFC
from cn_clip.clip import load_from_name

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seeds(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def save_results(args, history, best_preds, save_dir):
    """
    保存所有论文需要的实验结果
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 1. 保存配置参数 (Config)
    with open(os.path.join(save_dir, "config.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # 2. 保存训练日志 (用于画折线图: Epoch vs Loss/Acc/F1)
    with open(os.path.join(save_dir, "training_log.json"), 'w') as f:
        json.dump(history, f, indent=4)

    # 3. 保存最佳模型的详细预测结果 (用于画混淆矩阵、ROC曲线、Case分析)
    # best_preds 包含: [true_label, pred_label, prob_class_0, prob_class_1]
    df_preds = pd.DataFrame(best_preds)
    df_preds.to_csv(os.path.join(save_dir, "best_predictions.csv"), index=False)

    # 4. 生成并保存最佳模型的详细评估报告 (用于论文表格)
    y_true = df_preds['label'].values
    y_pred = df_preds['pred'].values

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # 生成详细报告
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)

    summary = {
        "Confusion Matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro F1": f1_score(y_true, y_pred, average='macro'),
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
    parser.add_argument("--shot", type=int, default=2)  # [cite: 168]
    parser.add_argument("--save_path", type=str, default="./checkpoints")
    args = parser.parse_args()

    set_seeds(args.seed)

    # 定义结果保存目录 (区分 Dataset, Shot, Seed)
    exp_name = f"{args.dataset_name}_{args.shot}shot_seed{args.seed}"
    result_dir = os.path.join("./paper_results", exp_name)  # 结果统一保存在 paper_results 文件夹

    print(f"🚀 Experiment: {exp_name}")
    print("Loading Chinese CLIP (Frozen)...")

    clip_model, preprocess = load_from_name("ViT-B-16", device=device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False  # 冻结 CLIP

    # 数据集准备
    train_dataset = FakeNews_Dataset(clip_model, preprocess, args.train_csv, args.img_path, args.dataset_name)
    test_dataset = FakeNews_Dataset(clip_model, preprocess, args.test_csv, args.img_path, args.dataset_name)

    # Few-shot 采样
    if args.dataset_name == 'ad':
        train_sampler = FewShotSampler_weibo(train_dataset, args.shot, args.seed)
        train_dataset = train_sampler.get_train_dataset()
    else:
        train_sampler = FewShotSampler_fakenewsnet(train_dataset, args.shot, args.seed)
        train_dataset, _ = train_sampler.get_train_val_datasets()

    print(f"Train Set Size: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 初始化模型 (这里如果用了 ACFC，记得把 CMA_Model 换成 CMA_Model_With_ACFC)
    cma_model = CMA_Model(feature_dim=512, num_classes=2).to(device)

    optimizer = AdamW(cma_model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_func = CrossEntropyLoss()

    best_acc = 0.0
    best_preds_data = []  # 用于保存最佳 Epoch 的预测详情

    # 用于记录训练过程
    history = {
        "epoch": [],
        "loss": [],
        "train_acc": [],
        "test_acc": [],
        "test_f1_macro": [],
        "test_f1_weighted": []
    }

    EPOCH = 20  # [cite: 165]

    for epoch in range(EPOCH):
        cma_model.train()
        total_loss = 0
        correct = 0
        total = 0

        for txt, img, label, mask in train_loader:
            txt, img, label, mask = txt.to(device), img.to(device), label.to(device), mask.to(device)

            # 获取维度: Batch, Slices, Channels, H, W
            B, S, C, H, W = img.shape

            # --- [关键步骤] 特征提取 ---
            with torch.no_grad():
                # 1. 展平 B 和 S 维度，让 CLIP 一次性处理所有切片
                img_flat = img.view(B * S, C, H, W)  # [B*S, 3, 224, 224]
                txt_flat = txt.view(B * S, -1)  # [B*S, 77]

                # 2. CLIP 提取
                img_feat_flat = clip_model.encode_image(img_flat)  # [B*S, 512]
                txt_feat_flat = clip_model.encode_text(txt_flat)  # [B*S, 512]

                # 3. 变回 [Batch, Slices, 512]
                img_feat = img_feat_flat.view(B, S, -1)
                txt_feat = txt_feat_flat.view(B, S, -1)

            # --- 前向传播 ---
            optimizer.zero_grad()

            # 传入 mask
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
        pred_probs_list = []  # 保存概率用于 ROC 曲线

        with torch.no_grad():
            for txt, img, label, mask in tqdm.tqdm(test_loader, desc="Testing"):
                txt, img, label, mask = txt.to(device), img.to(device), label.to(device), mask.to(device)

                # ... (特征提取和 View 变换代码保持不变) ...
                B, S, C, H, W = img.shape
                img_flat = img.view(B * S, C, H, W)
                txt_flat = txt.view(B * S, -1)
                img_feat_flat = clip_model.encode_image(img_flat)
                txt_feat_flat = clip_model.encode_text(txt_flat)
                img_feat = img_feat_flat.view(B, S, -1)
                txt_feat = txt_feat_flat.view(B, S, -1)

                # 前向传播
                logits = cma_model(txt_feat.float(), img_feat.float(), mask)

                # 计算概率
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=-1)

                # 【核心修改点】使用 append 而不是 extend，避免维度混乱
                test_labels.extend(label.cpu().numpy())
                pred_labels.extend(preds.cpu().numpy())
                pred_probs_list.append(probs.cpu().numpy())  # 把整个 Batch 的概率矩阵存进去

                # 【核心修改点】在循环外进行拼接
                # 将 list of arrays [ (64,2), (64,2), (10,2) ] -> big array (138, 2)
            if len(pred_probs_list) > 0:
                probs_np = np.concatenate(pred_probs_list, axis=0)
            else:
                probs_np = np.array([])

        # 计算各类指标
        curr_acc = accuracy_score(test_labels, pred_labels)
        macro_f1 = f1_score(test_labels, pred_labels, average='macro')
        weighted_f1 = f1_score(test_labels, pred_labels, average='weighted')

        # 更新日志
        history["epoch"].append(epoch + 1)
        history["loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(curr_acc)
        history["test_f1_macro"].append(macro_f1)
        history["test_f1_weighted"].append(weighted_f1)

        print(f"Test Accuracy: {curr_acc:.4f} | Macro F1: {macro_f1:.4f}")

        # 如果发现更好的模型
        if curr_acc > best_acc:
            best_acc = curr_acc
            print(f"New Best Accuracy: {best_acc:.4f}, Saving model & metrics...")

            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(cma_model.state_dict(), os.path.join(args.save_path, f"best_model_seed{args.seed}.pt"))

            # 2. 缓存预测数据
            # 此时 probs_np 已经是拼接好的 (N, 2) 数组了，可以直接切片
            best_preds_data = {
                "label": test_labels,
                "pred": pred_labels,
                "prob_0": probs_np[:, 0],  # 真实新闻概率
                "prob_1": probs_np[:, 1]  # 虚假新闻概率
            }

    print(f"Final Best Accuracy: {best_acc}")

    # 训练结束后，统一保存所有文件到 result_dir
    save_results(args, history, best_preds_data, result_dir)