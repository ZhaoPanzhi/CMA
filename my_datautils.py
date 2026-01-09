import os.path
from collections import defaultdict
import tqdm, random
from torch.utils.data import Dataset, Subset
import csv, clip, torch, cn_clip
import numpy as np
import pandas as pd
from PIL import Image
from cn_clip.clip import load_from_name


## 1 fake, 0 real
def get_base_id(img_name):
    # 根据你的 CSV 格式：图片名类似 "广告名_序号"
    # 该函数用于提取 "广告名" 作为分组依据
    img_name = str(img_name)
    if '_' in img_name:
        return img_name.rsplit('_', 1)[0]
    return img_name

class FakeNews_Dataset(Dataset):
    def __init__(self, model, preprocess, data_path, img_path, dataset_name, max_slices=8):

        self.img_path = img_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.preprocess = preprocess
        self.dataset_name = dataset_name
        self.max_slices = max_slices

        try:
            self.data = pd.read_csv(data_path, header=0, encoding='utf-8')
        except UnicodeDecodeError:
            # 如果 utf-8 也不行，尝试 gb18030 (兼容性更好的中文编码)
            self.data = pd.read_csv(data_path, header=0, encoding='gb18030')

        # [新增] 1. 生成 base_id 并分组
        self.data['base_id'] = self.data['image'].apply(get_base_id)
        self.groups = list(self.data.groupby('base_id'))  # 变成 [[base_id, df_group], ...]

        self.img_map = {}
        if os.path.exists(img_path):
            for filename in os.listdir(img_path):
                self.img_map[filename.lower()] = filename


    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        # [修改] 获取一组切片
        base_id, group = self.groups[idx]

        # 取该组第一个切片的标签（因为你已经修复了标签一致性）
        label = int(group.iloc[0]['label'])

        slice_imgs = []
        slice_txts = []

        # 遍历组内每一个切片
        for _, row in group.iterrows():
            img_name = str(row['image']) + ".jpg"  # 假设 CSV 里没后缀
            txt_raw = str(row['text'])

            # 空文本处理 (保留你之前的修复)
            if pd.isna(txt_raw) or txt_raw == "nan":
                txt_raw = ""

                # 图片读取
            target_name = img_name
            if os.path.exists(os.path.join(self.img_path, target_name)):
                final_img_name = target_name
            elif target_name.lower() in self.img_map:
                final_img_name = self.img_map[target_name.lower()]
            else:
                final_img_name = None  # 标记为缺失

            if final_img_name:
                img_tensor = self.preprocess(Image.open(os.path.join(self.img_path, final_img_name)))
            else:
                # 缺失图片补全黑
                img_tensor = torch.zeros(3, 224, 224)

            # 文本 Tokenize (CN-CLIP)
            txt_tensor = cn_clip.clip.tokenize(txt_raw).squeeze()  # [77]

            slice_imgs.append(img_tensor)
            slice_txts.append(txt_tensor)

            if len(slice_imgs) >= self.max_slices:
                break

        # [新增] Padding 补齐逻辑
        valid_num = len(slice_imgs)

        # 补图片 (0)
        while len(slice_imgs) < self.max_slices:
            slice_imgs.append(torch.zeros(3, 224, 224))
        # 补文本 (0)
        token_len = slice_txts[0].shape[0] if len(slice_txts) > 0 else 52
        while len(slice_txts) < self.max_slices:
            slice_txts.append(torch.zeros(token_len, dtype=torch.long))

        # 堆叠成 Tensor
        imgs_stack = torch.stack(slice_imgs)  # [Max_Slices, 3, 224, 224]
        txts_stack = torch.stack(slice_txts)  # [Max_Slices, 77]
        label = torch.tensor(label, dtype=torch.long)

        # 生成 Mask (1代表有效切片，0代表补齐的)
        mask = torch.zeros(self.max_slices)
        mask[:valid_num] = 1.0

        return txts_stack, imgs_stack, label, mask


class FewShotSampler_weibo:
    def __init__(self, dataset, few_shot_per_class, seed):
        self.dataset = dataset
        self.few_shot_per_class = few_shot_per_class
        self.seed = seed
    def get_train_dataset(self):
        indices_per_class = defaultdict(list)

        for idx in range(len(self.dataset)):
            # 直接从 dataset.groups 读取，避免触发 __getitem__ 图片加载
            base_id, group = self.dataset.groups[idx]
            label = int(group.iloc[0]['label'])
            indices_per_class[label].append(idx)

        train_indices = []
        for label, indices in indices_per_class.items():
            random.Random(self.seed).shuffle(indices)
            train_indices.extend(indices[:self.few_shot_per_class])

        return Subset(self.dataset, train_indices)
class FewShotSampler_fakenewsnet:
    def __init__(self, dataset, few_shot_per_class, seed):
        self.dataset = dataset
        self.few_shot_per_class = few_shot_per_class
        self.seed = seed
    def get_train_val_datasets(self):
        indices_per_class = defaultdict(list)
        for idx in range(len(self.dataset)):
            _, _, label = self.dataset[idx]
            indices_per_class[label.item()].append(idx)

        train_indices = []
        val_indices = []

        for label, indices in indices_per_class.items():
            random.Random(self.seed).shuffle(indices)
            train_indices.extend(indices[:self.few_shot_per_class])
            val_indices.extend(indices[self.few_shot_per_class:])

        train_dataset = Subset(self.dataset, train_indices)
        val_dataset = Subset(self.dataset, val_indices)

        return train_dataset, val_dataset

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