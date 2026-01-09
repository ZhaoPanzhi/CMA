import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter_Origin(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(Adapter_Origin, self).__init__()
        self.fc = torch.nn.Linear(512 * 2, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, feature_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.scale = feature_dim ** -0.5

    def forward(self, q_feat, k_feat, v_feat):
        Q = self.query(q_feat).unsqueeze(1)
        K = self.key(k_feat).unsqueeze(1)
        V = self.value(v_feat).unsqueeze(1)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_weights, V)
        return attn_output.squeeze(1)  # 返回 (Batch, 512)


class CMA_Model(nn.Module):
    def __init__(self, feature_dim=512, num_classes=2):
        super(CMA_Model, self).__init__()

        # 定义 Cross Attention 模块
        self.cross_att_mt = CrossAttention(feature_dim)  # Image guides Text
        self.cross_att_tm = CrossAttention(feature_dim)  # Text guides Image

        # 定义 5 个独立的 Linear Probing (对应论文 Figure 2 的 Linear Probing 部分)
        # [cite: 132] "each modality can be processed through the linear classifier MLP"
        self.lp_txt = nn.Linear(feature_dim, num_classes)
        self.lp_img = nn.Linear(feature_dim, num_classes)
        self.lp_cat = nn.Linear(feature_dim * 2, num_classes)  # Concat 维度是 1024
        self.lp_mt = nn.Linear(feature_dim, num_classes)
        self.lp_tm = nn.Linear(feature_dim, num_classes)

        # 定义 Meta-Linear Probing (对应论文 Eq 5 和 Figure 2 最右侧)
        # 输入是上述 5 个分类器的输出 (logits) 的拼接
        # 5 个分类器 * num_classes
        self.meta_classifier = nn.Linear(5 * num_classes, num_classes)

    def forward(self, txt_feat, img_feat):
        # 1. 准备基础特征
        # 论文 [cite: 123] 要求对拼接前的特征做 L2 Normalize
        f_t = F.normalize(txt_feat, dim=-1)
        f_m = F.normalize(img_feat, dim=-1)

        # 2. 构建 5 种特征视角的表示 (Representations)
        # View 1: Text only
        feat_t = f_t
        # View 2: Image only
        feat_m = f_m
        # View 3: Concatenation [cite: 123]
        feat_c = torch.cat((f_t, f_m), dim=-1)
        # View 4: Cross-Attn (Image Query, Text Key/Val) -> f_mt
        feat_mt = self.cross_att_mt(f_m, f_t, f_t)
        # View 5: Cross-Attn (Text Query, Image Key/Val) -> f_tm
        feat_tm = self.cross_att_tm(f_t, f_m, f_m)

        # 3. 第一层: Linear Probing (获取 5 组 Logits)
        logits_t = self.lp_txt(feat_t)
        logits_m = self.lp_img(feat_m)
        logits_c = self.lp_cat(feat_c)
        logits_mt = self.lp_mt(feat_mt)
        logits_tm = self.lp_tm(feat_tm)

        # 4. 第二层: Meta-Linear Probing
        # 论文 [cite: 134] 公式 (5): MLP(ft + fm + fc + fmt + ftm)
        # 这里的 "+" 代表 concatenate
        meta_input = torch.cat((logits_t, logits_m, logits_c, logits_mt, logits_tm), dim=-1)

        final_logits = self.meta_classifier(meta_input)

        return final_logits