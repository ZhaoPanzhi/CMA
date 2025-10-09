import os, os.path, json, time, random, argparse
import numpy as np
import torch
import tqdm
import clip
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast, GradScaler
from cn_clip.clip import load_from_name

import matplotlib
matplotlib.use("Agg")  # 后端设为非交互，便于服务器/无显示环境保存图
import matplotlib.pyplot as plt
import itertools

from my_datautils import FakeNews_Dataset, FewShotSampler_fakenewsnet, FewShotSampler_weibo
from mymodels import Adapter_Origin, Adapter_V1
from mymodels import FEATHead

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seeds(seed: int = 42, deterministic: bool = True):
    """Make everything as reproducible as possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def format_seconds(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def plot_confusion(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, out_path=None):
    """保存混淆矩阵图片"""
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-12)

    plt.figure(figsize=(6, 5), dpi=160)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2. if cm.size > 0 else 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=9)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()


def run_eval(model, adapter, dataloader, num_classes=2, use_amp=True):
    """
    在验证/测试集上评估，返回：
      - classification_report (dict)
      - confusion_matrix (np.ndarray)
      - y_true (np.ndarray)
      - y_pred (np.ndarray)
      - eval_speed (dict)
    """
    adapter.eval()
    all_preds, all_labels = [], []
    step_times = []

    with torch.no_grad():
        pbar = tqdm.tqdm(dataloader, desc="Eval", leave=False)
        for txt, img, label in pbar:
            # ✅ 确保数据在 GPU
            txt = txt.to(device, non_blocking=True)
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            start = time.time()
            with autocast(enabled=use_amp):
                # CLIP 编码
                img_feat_1 = model.encode_image(img)
                txt_feat_1 = model.encode_text(txt)

                # L2 norm
                img_feat = img_feat_1 / img_feat_1.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat_1 / txt_feat_1.norm(dim=-1, keepdim=True)

                # 拼接特征
                all_feat = torch.cat((img_feat, txt_feat), dim=-1).to(device, torch.float32)

                # Adapter 推理
                _, _, eval_logits = adapter(
                    txt_feat_1.to(device, torch.float32),
                    img_feat_1.to(device, torch.float32),
                    all_feat
                )

                preds = torch.argmax(torch.softmax(eval_logits, dim=1), dim=-1)

            step_times.append(time.time() - start)

            # 收集预测与标签
            all_preds.append(preds.cpu().numpy())
            all_labels.append(label.cpu().numpy())

    # 拼接结果
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    labels = list(range(num_classes))
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    eval_speed = {
        "avg_step_time_sec": float(np.mean(step_times)) if step_times else 0.0,
        "steps": len(step_times)
    }
    return report, cm, y_true, y_pred, eval_speed


def run_eval_feat(model, feat_head, support_feats, support_labels, dataloader, num_classes=2, use_amp=True):
    feat_head.eval()
    all_preds, all_labels = [], []
    step_times = []
    with torch.no_grad():
        pbar = tqdm.tqdm(dataloader, desc="Eval(FEAT)", leave=False)
        for txt, img, label in pbar:
            txt = txt.to(device, non_blocking=True)
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            start = time.time()
            with autocast(enabled=use_amp):
                img_feat_1 = model.encode_image(img)
                txt_feat_1 = model.encode_text(txt)
                img_feat = img_feat_1 / img_feat_1.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat_1 / txt_feat_1.norm(dim=-1, keepdim=True)
                q_feats = torch.cat((img_feat, txt_feat), dim=-1).to(device, torch.float32)

                logits, class_order = feat_head(support_feats, support_labels, q_feats)
                preds = torch.argmax(torch.softmax(logits, dim=1), dim=-1)

                # 把 class_order 的索引还原成真实标签
                class_list = [int(c.item()) for c in class_order]
                mapped_preds = torch.tensor([class_list[int(p.item())] for p in preds],
                                            device=device, dtype=torch.long)

            step_times.append(time.time() - start)
            all_preds.append(mapped_preds.cpu().numpy())
            all_labels.append(label.cpu().numpy())

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    labels = list(range(num_classes))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    eval_speed = {
        "avg_step_time_sec": float(np.mean(step_times)) if step_times else 0.0,
        "steps": len(step_times)
    }
    return report, cm, y_true, y_pred, eval_speed


def save_epoch_metrics(csv_path, epoch, train_loss, report, lr, train_speed, eval_speed):
    """把每个 epoch 的指标追加写入 CSV"""
    import csv
    headers = [
        "epoch", "train_loss", "lr",
        "accuracy", "macro_f1", "macro_precision", "macro_recall",
        "micro_f1", "micro_precision", "micro_recall",
        "train_avg_step_time_sec", "train_steps",
        "eval_avg_step_time_sec", "eval_steps"
    ]
    row = [
        epoch,
        f"{train_loss:.6f}",
        f"{lr:.6e}",
        f"{report.get('accuracy', 0):.6f}",
        f"{report.get('macro avg', {}).get('f1-score', 0):.6f}",
        f"{report.get('macro avg', {}).get('precision', 0):.6f}",
        f"{report.get('macro avg', {}).get('recall', 0):.6f}",
        f"{report.get('weighted avg', {}).get('f1-score', 0):.6f}",  # 这里保留 weighted 以供对比
        f"{report.get('weighted avg', {}).get('precision', 0):.6f}",
        f"{report.get('weighted avg', {}).get('recall', 0):.6f}",
        f"{train_speed.get('avg_step_time_sec', 0):.6f}",
        f"{train_speed.get('steps', 0)}",
        f"{eval_speed.get('avg_step_time_sec', 0):.6f}",
        f"{eval_speed.get('steps', 0)}",
    ]

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)

def build_support_cache(dataloader_or_dataset, model, use_amp=True):
    """
    从 few-shot 训练子集（Subset/FewShotSampler 输出）提取 support 的 CMA 融合特征 all_feat 与标签。
    返回:
      support_feats: [Ns, 1024]  Tensor on device
      support_labels: [Ns]       Tensor on device (原始类标)
    """
    from torch.utils.data import DataLoader
    if isinstance(dataloader_or_dataset, DataLoader):
        loader = dataloader_or_dataset
    else:
        loader = DataLoader(dataloader_or_dataset, batch_size=512, shuffle=False,
                            num_workers=0, pin_memory=torch.cuda.is_available())

    feats, labels = [], []
    with torch.no_grad():
        for txt, img, label in tqdm.tqdm(loader, desc="Build support"):
            txt = txt.to(device, non_blocking=True)
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                img_feat = model.encode_image(img)
                txt_feat = model.encode_text(txt)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                all_feat = torch.cat((img_feat, txt_feat), dim=-1).to(device, torch.float32)
            feats.append(all_feat)
            labels.append(label)
    support_feats = torch.cat(feats, dim=0)
    support_labels = torch.cat(labels, dim=0)
    return support_feats, support_labels



def main():
    parser = argparse.ArgumentParser(description="help")
    parser.add_argument("--seed", type=int, required=True, help="seed number")
    parser.add_argument("--dataset_name", type=str, required=True, help="weibo, politifact, gossipcop")
    parser.add_argument("--train_csv", type=str, required=True, help="train csv")
    parser.add_argument("--test_csv", type=str, required=True, help="test csv (for weibo); or will be split by sampler")
    parser.add_argument("--img_path", type=str, required=True, help="img root")
    parser.add_argument("--shot", type=int, required=True, help='few-shots')
    parser.add_argument("--save_path", type=str, required=True, help="dir to save outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)  # Windows 建议 0
    parser.add_argument("--amp", action="store_true", help="use mixed precision (FP16)")
    parser.add_argument("--use_feat", action="store_true", help="use FEAT head with CMA fused features")
    parser.add_argument("--feat_heads", type=int, default=4)
    parser.add_argument("--feat_layers", type=int, default=1)
    args = parser.parse_args()

    set_seeds(args.seed)
    ensure_dir(args.save_path)

    # 记录本次配置
    with open(os.path.join(args.save_path, "train_config.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps(vars(args), indent=2, ensure_ascii=False))

    print(f"Device: {device}")
    print(f"SEED: {args.seed} | DATASET: {args.dataset_name} | SHOT: {args.shot}")
    print(f"Save to: {args.save_path}\n")

    data_name = args.dataset_name
    # ===== Data & Model =====
    if data_name == "weibo":
        print("Loading Chinese CLIP (cn_clip) .....")
        model, preprocess = load_from_name("ViT-B-16", device=device)
        train_dataset = FakeNews_Dataset(model, preprocess, args.train_csv, args.img_path, data_name)
        test_dataset = FakeNews_Dataset(model, preprocess, args.test_csv, args.img_path, data_name)

        train_sampler = FewShotSampler_weibo(train_dataset, args.shot, args.seed)
        train_dataset = train_sampler.get_train_dataset()
        torch.manual_seed(args.seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=lambda _: np.random.seed(args.seed),
            pin_memory=torch.cuda.is_available()
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available()
        )
    else:
        print("Loading OpenAI CLIP .....")
        model, preprocess = clip.load('ViT-B/32', device, jit=False)
        train_dataset = FakeNews_Dataset(model, preprocess, args.train_csv, args.img_path, data_name)

        sampler = FewShotSampler_fakenewsnet(train_dataset, args.shot, args.seed)
        train_dataset, test_dataset = sampler.get_train_val_datasets()
        torch.manual_seed(args.seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=lambda _: np.random.seed(args.seed),
            pin_memory=torch.cuda.is_available()
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    # ===== Adapter & Optim =====
    # adapter = Adapter_Origin(num_classes=2).to(device)
    if args.use_feat:
        # 使用 FEAT：优化 FEAT 头参数（CMA/CLIP 冻结为特征提取）
        feat_head = FEATHead(in_dim=1024, num_heads=args.feat_heads, depth=args.feat_layers).to(device)
        optimizer = AdamW(list(feat_head.parameters()), lr=args.lr, eps=args.eps)
        loss_func = CrossEntropyLoss()
        model_to_save = feat_head
    else:
        adapter = Adapter_V1(num_classes=2).to(device)
        optimizer = AdamW(adapter.parameters(), lr=args.lr, eps=args.eps)
        loss_func = CrossEntropyLoss()
        model_to_save = adapter
    scaler = GradScaler(enabled=args.amp)

    # ===== Meta Info =====
    EPOCHS = args.epochs
    best_acc = 0.0
    patience = 3
    patience_count = 0

    # 统计参数量
    num_params = sum(p.numel() for p in model_to_save.parameters())
    num_train = len(train_loader.dataset) if hasattr(train_loader, "dataset") else -1
    num_test = len(test_loader.dataset) if hasattr(test_loader, "dataset") else -1
    print(f"Adapter params: {num_params:,}")
    print(f"Train size: {num_train}, Test size: {num_test}\n")

    metrics_csv = os.path.join(args.save_path, "metrics_epoch.csv")

    # ===== Train Loop =====
    for epoch in range(1, EPOCHS + 1):
        if args.use_feat:
            feat_head.train()
        else:
            adapter.train()
        epoch_loss, step_times = 0.0, []
        start_epoch = time.time()
        pbar = tqdm.tqdm(train_loader, desc=f"Train | Epoch {epoch}/{EPOCHS}")

        for txt, img, label in pbar:
            txt = txt.to(device, non_blocking=True)
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            step_start = time.time()
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=args.amp):
                img_feat_0 = model.encode_image(img)
                txt_feat_0 = model.encode_text(txt)
                img_feat = img_feat_0 / img_feat_0.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat_0 / txt_feat_0.norm(dim=-1, keepdim=True)
                all_feat = torch.cat((img_feat, txt_feat), dim=-1).to(device, torch.float32)  # [B, 1024]

            if args.use_feat:
                # -------- FEAT 训练（从当前 batch 中切出 support/query） --------
                # 期望 batch 内包含至少每类 K=shot 的样本
                K = args.shot
                labels_np = label.detach().cpu().numpy()
                classes = np.unique(labels_np)

                support_idx = []
                remain_idx = list(range(len(labels_np)))
                for c in classes:
                    idx_c = np.where(labels_np == c)[0].tolist()
                    if len(idx_c) < K:
                        # 如果本 batch 不足以形成完整 episode，就跳过这个 step
                        # 也可以累积到下个 batch，这里先简单跳过
                        support_idx = []
                        break
                    support_idx.extend(idx_c[:K])
                if len(support_idx) == 0:
                    # 跳过该 step，不计梯度
                    continue

                query_idx = sorted(set(remain_idx) - set(support_idx))
                support_idx = torch.tensor(support_idx, dtype=torch.long, device=device)
                query_idx = torch.tensor(query_idx, dtype=torch.long, device=device)

                s_feats = all_feat.index_select(0, support_idx)  # [Ns, D]
                s_labels = label.index_select(0, support_idx)  # [Ns]
                q_feats = all_feat.index_select(0, query_idx)  # [Nq, D]
                q_labels = label.index_select(0, query_idx)  # [Nq]

                logits, class_order = feat_head(s_feats, s_labels, q_feats)  # [Nq, C], [C]
                # 把原始标签映射到 [0..C-1]
                label_map = {int(c.item()): i for i, c in enumerate(class_order)}
                target = torch.tensor([label_map[int(x.item())] for x in q_labels], device=device, dtype=torch.long)
                loss = loss_func(logits, target)
            else:
                # -------- 原 Adapter 路径 --------
                with autocast(enabled=args.amp):
                    _, _, logits = adapter(
                        txt_feat_0.to(device, torch.float32),
                        img_feat_0.to(device, torch.float32),
                        all_feat
                    )
                    loss = loss_func(logits, label)

            # backward
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            step_time = time.time() - step_start
            step_times.append(step_time)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "step_s": f"{step_time:.3f}"})

        train_avg_loss = epoch_loss / max(1, len(train_loader))
        train_speed = {
            "avg_step_time_sec": float(np.mean(step_times)) if step_times else 0.0,
            "steps": len(step_times)
        }
        epoch_time = time.time() - start_epoch

        # ===== Eval =====
        print("Start Eval ...")
        if args.use_feat:
            # 评测时的 support = few-shot 训练子集（train_loader.dataset）
            # 注意：weibo 分支里 train_dataset 已经是 FewShotSampler_weibo 产物（Subset）
            support_feats, support_labels = build_support_cache(train_loader.dataset, model, use_amp=args.amp)
            report, cm, y_true, y_pred, eval_speed = run_eval_feat(
                model=model,
                feat_head=feat_head,
                support_feats=support_feats,
                support_labels=support_labels,
                dataloader=test_loader,
                num_classes=2,
                use_amp=args.amp
            )
        else:
            report, cm, y_true, y_pred, eval_speed = run_eval(
                model=model,
                adapter=adapter,
                dataloader=test_loader,
                num_classes=2,
                use_amp=args.amp
            )
        acc = float(report.get("accuracy", 0.0))
        macro_f1 = float(report.get("macro avg", {}).get("f1-score", 0.0))

        # 打印关键指标
        print(f"[Epoch {epoch}] "
              f"train_loss={train_avg_loss:.4f} | "
              f"acc={acc:.4f} | macro_f1={macro_f1:.4f} | "
              f"epoch_time={format_seconds(epoch_time)} | "
              f"train_step={train_speed['avg_step_time_sec']:.3f}s | "
              f"eval_step={eval_speed['avg_step_time_sec']:.3f}s")

        # 保存混淆矩阵（原始 & 归一化）
        plot_confusion(
            cm, classes=[str(i) for i in range(2)], normalize=False,
            title=f"Confusion Matrix (Epoch {epoch})",
            out_path=os.path.join(args.save_path, f"confusion_matrix_epoch{epoch:02d}.png")
        )
        plot_confusion(
            cm, classes=[str(i) for i in range(2)], normalize=True,
            title=f"Confusion Matrix Normalized (Epoch {epoch})",
            out_path=os.path.join(args.save_path, f"confusion_matrix_norm_epoch{epoch:02d}.png")
        )

        # 逐 epoch 写 CSV
        current_lr = optimizer.param_groups[0]["lr"]
        save_epoch_metrics(
            csv_path=metrics_csv,
            epoch=epoch,
            train_loss=train_avg_loss,
            report=report,
            lr=current_lr,
            train_speed=train_speed,
            eval_speed=eval_speed
        )

        # ===== Save Best =====
        if acc > best_acc:
            best_acc = acc
            patience_count = 0
            print(f"New best at epoch {epoch}: acc={best_acc:.4f}. Saving model & artifacts...")

            # 模型
            if args.use_feat:
                model_path = os.path.join(args.save_path,
                                          f"seed{args.seed}_FEAT_shot{args.shot}@{args.dataset_name}_best.pt")
            else:
                model_path = os.path.join(args.save_path,
                                          f"seed{args.seed}_adapter_shot{args.shot}@{args.dataset_name}_best.pt")
            torch.save(model_to_save.state_dict(), model_path)

            # 最佳指标快照
            best_metrics = {
                "epoch": epoch,
                "accuracy": report.get("accuracy", 0.0),
                "macro_avg": report.get("macro avg", {}),
                "weighted_avg": report.get("weighted avg", {}),
                "classes": {k: v for k, v in report.items() if k not in ["accuracy", "macro avg", "weighted avg"]},
            }
            with open(os.path.join(args.save_path, "best_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(best_metrics, f, ensure_ascii=False, indent=2)

            # 保存该轮预测明细
            import pandas as pd
            pd.DataFrame({"label": y_true, "pred": y_pred}).to_csv(
                os.path.join(args.save_path, "predictions_best.csv"),
                index=False, encoding="utf-8"
            )
        else:
            patience_count += 1
            print(f"No improvement. patience={patience_count}/{patience}")

        if patience_count >= patience:
            print("Early stopping triggered.")
            break

    print(f"Best accuracy: {best_acc:.4f}")
    # 兼容你原来的文本记录
    with open(os.path.join(args.save_path, f"seed{args.seed}_shot{args.shot}@{args.dataset_name}.txt"), 'w') as outf:
        outf.write(str(round(best_acc, 4)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting gracefully.")
