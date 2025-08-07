# export_weibo_csv.py
from src.process_data_weibo import get_data
import os, csv

# 加载处理后的数据
train, val, test = get_data(text_only=False)

def export(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id', 'text', 'label', 'image'])  # 正确字段顺序
        for i, (txt, imgid, lbl) in enumerate(zip(data['original_post'], data['image_id'], data['label'])):
            img = imgid  # 不加 .jpg 后缀，保存文件名
            img_path = os.path.join('../datasets/weibo/all_images', img + '.jpg')  # 检查图像时加 .jpg

            # 跳过没有有效图像路径的样本
            if not os.path.exists(img_path):
                print(f"[WARNING] Image not found for ID {i}: {img_path}")  # 打印警告
                continue  # 跳过该样本

            w.writerow([i, txt.strip(), lbl, img])  # 只有有效图像才写入，不加 .jpg 后缀

export(train, '../datasets/weibo/weibo_train.csv')
export(test, '../datasets/weibo/weibo_test.csv')
