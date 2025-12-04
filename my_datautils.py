import os.path
from collections import defaultdict
import tqdm, random
from torch.utils.data import Dataset, Subset
import csv, clip, torch, cn_clip
import numpy as np
import pandas as pd
from PIL import Image
from cn_clip.clip import load_from_name
import glob


## 1 fake, 0 real

class FakeNews_Dataset(Dataset):
    def __init__(self, model, preprocess, data_path, img_path, dataset_name):

        self.img_path = img_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.preprocess = preprocess
        self.dataset_name = dataset_name
        self.data = pd.read_csv(data_path, encoding="utf-8")


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text_raw = row["text"]
        # --- 修复 NaN 文本（核心补丁） ---
        if isinstance(text_raw, float) and pd.isna(text_raw):
            text = ""  # 将 NaN 转为空字符串
        else:
            text = str(text_raw)
        # -----------------------------------
        label = int(row["label"])
        img_name = str(row["image"]).strip()

        # 如果 csv 里没有后缀，加上 .jpg
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            img_name = img_name + ".jpg"

        img_path = os.path.join(self.img_path, img_name)

        # ✅ 如果文件不存在，尝试大小写无关匹配
        if not os.path.exists(img_path):
            # 用 glob 搜索所有可能的大小写组合
            candidates = glob.glob(os.path.join(self.img_path, "*"))
            lower_map = {os.path.basename(c).lower(): c for c in candidates}
            if os.path.basename(img_name).lower() in lower_map:
                img_path = lower_map[os.path.basename(img_name).lower()]
            else:
                raise FileNotFoundError(f"图片找不到: {img_path}")

        img = self.preprocess(Image.open(img_path))

        if self.dataset_name in ["weibo", "ad"]:
            txt = cn_clip.clip.tokenize(text).squeeze()
        else:
            txt = clip.tokenize(text, truncate=True).squeeze()

        label = torch.as_tensor(label, dtype=torch.long)

        return txt, img, label


class FewShotSampler_weibo:
    def __init__(self, dataset, few_shot_per_class, seed, resample=False):
        self.dataset = dataset
        self.few_shot_per_class = few_shot_per_class
        self.seed = seed
        self.resample = resample
    def get_train_dataset(self):
        indices_per_class = defaultdict(list)
        for idx in range(len(self.dataset)):
            _, _, label = self.dataset[idx]
            indices_per_class[label.item()].append(idx)

        train_indices = []

        for label, indices in indices_per_class.items():
            if getattr(self, "resample", False):
                # 开启随机采样模式：每次运行都不同
                random.shuffle(indices)
            else:
                # 固定模式：可复现
                random.Random(self.seed).shuffle(indices)
            train_indices.extend(indices[:self.few_shot_per_class])

        train_dataset = Subset(self.dataset, train_indices)

        return train_dataset

# class FewShotSampler_ad:
#     """
#     适用于广告违规识别（0/1 二分类）的 Few-Shot Sampler。
#     - 支持 K-shot（few_shot_per_class）
#     - 支持 resample（每次运行随机抽样）
#     - 与 FEAT 完美兼容
#     """
#
#     def __init__(self, dataset, few_shot_per_class, seed=42, resample=False):
#         self.dataset = dataset
#         self.few_shot_per_class = few_shot_per_class
#         self.seed = seed
#         self.resample = resample
#
#     def get_train_dataset(self):
#         from pathlib import Path
#
#         # ====== 1. few-shot 子集保存路径（按 shot + seed 唯一） ======
#         save_dir = Path("saved_fewshot_indices")
#         save_dir.mkdir(parents=True, exist_ok=True)
#
#         index_file = save_dir / f"ad_shot{self.few_shot_per_class}_seed{self.seed}.npy"
#
#         # ====== 2. 如果文件已存在 → 直接加载固定子集（核心） ======
#         if index_file.exists():
#             print(f"[FewShotSampler_ad] 加载固定 few-shot 子集: {index_file}")
#             train_indices = np.load(index_file).tolist()
#             return Subset(self.dataset, train_indices)
#
#         print(f"[FewShotSampler_ad] 第一次创建 few-shot 子集并保存到: {index_file}")
#
#         # ====== 3. 正常按 seed 抽样（第一次运行） ======
#         indices_per_class = defaultdict(list)
#         for idx in range(len(self.dataset)):
#             _, _, label = self.dataset[idx]
#             indices_per_class[int(label.item())].append(idx)
#
#         rng = random.Random(self.seed)  # 固定随机源
#         train_indices = []
#
#         for label, indices in indices_per_class.items():
#             rng.shuffle(indices)  # 使用 seed = 固定顺序
#
#             shot = self.few_shot_per_class
#             if len(indices) < shot:
#                 print(f"[Warning] 类 {label} 样本不足 {shot}，仅抽取 {len(indices)} 个.")
#                 train_indices.extend(indices)
#             else:
#                 train_indices.extend(indices[:shot])
#
#         # ====== 4. 保存子集，今后的训练永远复用 ======
#         np.save(index_file, np.array(train_indices))
#         print(f"[FewShotSampler_ad] few-shot 子集保存成功: {index_file}")
#
#         return Subset(self.dataset, train_indices)

class FewShotSampler_ad:
    def __init__(self, dataset, few_shot_per_class, seed=42, resample=False):
        self.dataset = dataset
        self.few_shot_per_class = few_shot_per_class
        self.seed = seed
        self.resample = resample

    def get_train_dataset(self):
        from collections import defaultdict
        import random

        random.seed(self.seed)
        indices_per_class = defaultdict(list)

        # 按类别收集
        for idx in range(len(self.dataset)):
            _, _, label = self.dataset[idx]
            indices_per_class[int(label.item())].append(idx)

        train_indices = []

        for cls, idx_list in indices_per_class.items():
            # 随机打乱
            random.shuffle(idx_list)

            # 如果不够 shot，则重复补齐，保证每类数量一致
            if len(idx_list) < self.few_shot_per_class:
                repeat = self.few_shot_per_class - len(idx_list)
                selected = idx_list + random.choices(idx_list, k=repeat)
            else:
                selected = idx_list[:self.few_shot_per_class]

            train_indices.extend(selected)

        # 打印调试信息
        labels = [int(self.dataset[i][2].item()) for i in train_indices]
        print(f"[FewShotSampler] shot={self.few_shot_per_class}  labels={labels}")

        return Subset(self.dataset, train_indices)



class FewShotSampler_fakenewsnet:
    def __init__(self, dataset, few_shot_per_class, seed, resample=False):
        self.dataset = dataset
        self.few_shot_per_class = few_shot_per_class
        self.seed = seed
        self.resample = resample

    def get_train_val_datasets(self):
        indices_per_class = defaultdict(list)
        for idx in range(len(self.dataset)):
            _, _, label = self.dataset[idx]
            indices_per_class[label.item()].append(idx)

        train_indices, val_indices = [], []

        for label, indices in indices_per_class.items():
            if self.resample:
                random.shuffle(indices)
            else:
                random.Random(self.seed).shuffle(indices)
            train_indices.extend(indices[:self.few_shot_per_class])
            val_indices.extend(indices[self.few_shot_per_class:])

        return Subset(self.dataset, train_indices), Subset(self.dataset, val_indices)

def sim_cal(txt_path, img_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_from_name("ViT-B-16", device=device,
                                       download_root='~/PycharmProjects/local_models/clip-chinese')
    model.eval()
    cos = torch.nn.CosineSimilarity()
    all_txt, all_img, all_label = [], [], []
    with open(txt_path, 'r') as inf:
        content = inf.readlines()
        for i in tqdm.tqdm(range(0, len(content), 3)):
            three_lines = content[i:i + 3]
            if len(three_lines[2].strip()) != 0:  # remove empty text
                urls = three_lines[1].split("|")
                txt = cn_clip.clip.tokenize(three_lines[2].strip()).to(device)
                txt_emb = model.encode_text(txt)
                img_final = None
                tmp_sim = 0
                for item in urls:
                    if item != 'null\n':
                        url = item.split("/")[-1]
                        if os.path.exists(img_path + url):
                            img = preprocess(Image.open(img_path + url)).unsqueeze(0).to(device)
                            img_emb = model.encode_image(img)
                            sim = cos(img_emb, txt_emb).cpu().detach().numpy()
                            if sim >= tmp_sim:
                                tmp_sim = sim
                                img_final = url
                if img_final is not None:
                    img_final = img_final.split(".")[0]
                    all_img.append(img_final)
                    all_txt.append(three_lines[2].strip())
                    if img_path.split("/")[-2] == "rumor_images":
                        all_label.append("1")
                    else:
                        all_label.append("0")

    return all_txt, all_img, all_label

def weibo_2_csv(fake_txt, fake_img, real_txt, real_img, out_path):
    ftxt, fimg, flabel = sim_cal(fake_txt, fake_img)
    rtxt, rimg, rlabel = sim_cal(real_txt, real_img)

    txt = ftxt + rtxt
    img = fimg + rimg
    label = flabel + rlabel
    with open(out_path, 'w', newline='') as outf:
        csv_writer = csv.writer(outf)
        for l in range(len(label)):
            csv_writer.writerow([txt[l], img[l], label[l]])

def load_from_csv(csv_path, img_path):
    all_txt, all_img, all_label = [], [], []
    with open(csv_path, 'r') as inf:
        data = csv.reader(inf)
        for line in data:
            img = line[2] + ".jpg"
            label = line[3]
            txt = line[1]
            all_img.append(img_path + img)
            all_txt.append(txt)
            all_label.append(label)

    return all_txt, all_img, all_label