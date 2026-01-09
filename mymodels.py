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


class SliceAttentionFusion(nn.Module):
    def __init__(self, feature_dim=512):
        super(SliceAttentionFusion, self).__init__()
        # 简单的 Attention 网络：输入特征 -> 输出权重 score
        self.attn_fc = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x, mask):
        # x: [Batch, Slices, Dim]
        # mask: [Batch, Slices] (1有效, 0无效)

        # 计算每个切片的重要性分数
        scores = self.attn_fc(x).squeeze(-1)  # [Batch, Slices]

        # Mask 处理：将无效切片的分数设为负无穷，这样 Softmax 后权重为 0
        # 注意 mask 需要匹配 scores 的 device
        scores = scores.masked_fill(mask == 0, -1e9)

        # 归一化权重
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # [Batch, Slices, 1]

        # 加权求和融合
        fused_feat = torch.sum(x * weights, dim=1)  # [Batch, Dim]
        return fused_feat

class CMA_Model(nn.Module):
    def __init__(self, feature_dim=512, num_classes=2):
        super(CMA_Model, self).__init__()

        # 定义 Cross Attention 模块
        self.cross_att_mt = CrossAttention(feature_dim)  # Image guides Text
        self.cross_att_tm = CrossAttention(feature_dim)  # Text guides Image

        # [新增] 定义 5 个融合器 (针对 5 种 View)
        # 1. Text View
        self.fusion_t = SliceAttentionFusion(feature_dim)
        # 2. Image View
        self.fusion_m = SliceAttentionFusion(feature_dim)
        # 3. Concat View (维度是 1024)
        self.fusion_c = SliceAttentionFusion(feature_dim * 2)
        # 4. Cross T->I View
        self.fusion_mt = SliceAttentionFusion(feature_dim)
        # 5. Cross I->T View
        self.fusion_tm = SliceAttentionFusion(feature_dim)

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

    def forward(self, txt_feat, img_feat, mask):
        # 1. 准备基础特征
        # 论文 [cite: 123] 要求对拼接前的特征做 L2 Normalize
        f_t = F.normalize(txt_feat, dim=-1)
        f_m = F.normalize(img_feat, dim=-1)

        # 2. 构建 5 种特征视角的表示 (Representations)
        # View 1: Text only
        feat_t = f_t
        # View 2: Image only
        feat_m = f_m

        feat_c = torch.cat((f_t, f_m), dim=-1)

        B, S, D = f_t.shape
        flat_t = f_t.view(B * S, D)
        flat_m = f_m.view(B * S, D)

        # View 3: Concatenation [cite: 123]
        # View 4: Cross-Attn (Image Query, Text Key/Val) -> f_mt
        flat_mt = self.cross_att_mt(flat_m, flat_t, flat_t)
        flat_tm = self.cross_att_tm(flat_t, flat_m, flat_m)

        feat_mt = flat_mt.view(B, S, D)
        feat_tm = flat_tm.view(B, S, D)

        # 3. [新增] 多切片融合 (Multi-slice Fusion)
        # 将 [Batch, Slices, Dim] -> [Batch, Dim]
        final_t = self.fusion_t(feat_t, mask)
        final_m = self.fusion_m(feat_m, mask)
        final_c = self.fusion_c(feat_c, mask)
        final_mt = self.fusion_mt(feat_mt, mask)
        final_tm = self.fusion_tm(feat_tm, mask)

        # 3. 第一层: Linear Probing (获取 5 组 Logits)
        logits_t = self.lp_txt(final_t)
        logits_m = self.lp_img(final_m)
        logits_c = self.lp_cat(final_c)
        logits_mt = self.lp_mt(final_mt)
        logits_tm = self.lp_tm(final_tm)

        # 4. 第二层: Meta-Linear Probing
        # 论文 [cite: 134] 公式 (5): MLP(ft + fm + fc + fmt + ftm)
        # 这里的 "+" 代表 concatenate
        meta_input = torch.cat((logits_t, logits_m, logits_c, logits_mt, logits_tm), dim=-1)

        final_logits = self.meta_classifier(meta_input)

        return final_logits