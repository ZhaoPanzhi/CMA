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
        text = row["text"]
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

        if self.dataset_name == "weibo":
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