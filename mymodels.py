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


# [新增] SADG 门控模块
class SADG_Gating(nn.Module):
    def __init__(self, feature_dim=512, kernel_size=3):
        super(SADG_Gating, self).__init__()

        # 1. 相似度投影层：将标量相似度映射为特征向量
        self.sim_proj = nn.Sequential(
            nn.Linear(1, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, feature_dim)
        )

        # 2. 交互层：使用 1D 卷积代替全连接层 (ECA-Net思想)
        # 输入维度 (Batch, 1, 512) -> 输出 (Batch, 1, 512)
        # 这种卷积是在特征维度上进行的，参数极少
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, txt_feat, img_feat):
        # 输入: (N, 512) 这里 N = Batch * Slices

        # 1. 计算余弦相似度 (-1 到 1)
        sim = F.cosine_similarity(txt_feat, img_feat, dim=-1, eps=1e-8).unsqueeze(1)  # (N, 1)

        # 2. 注入相似度先验
        sim_feat = self.sim_proj(sim)  # (N, 512)

        # 3. 特征融合 (Text + Image + Sim)
        combined = txt_feat + img_feat + sim_feat

        # 4. 1D 卷积交互生成门控
        # unsqueeze(1) 变成 (N, 1, 512) 适应 Conv1d
        gate_logits = self.conv(combined.unsqueeze(1)).squeeze(1)  # (N, 512)
        weights = self.sigmoid(gate_logits)

        # 5. 差异化门控
        # 对 Text: 保留权重
        feat_txt_gated = txt_feat * weights

        # 对 Image: 额外乘上相似度系数 (如果是图文不符的切片，强制压低权重)
        sim_score = (sim + 1.0) / 2.0  # 归一化到 0~1
        feat_img_gated = img_feat * weights * sim_score

        return feat_txt_gated, feat_img_gated

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

        # --- [修改点 1] 初始化 SADG 模块 ---
        self.sadg = SADG_Gating(feature_dim)

        # 原始 CMA 组件
        self.cross_att_mt = CrossAttention(feature_dim)
        self.cross_att_tm = CrossAttention(feature_dim)

        # 多切片融合组件
        self.fusion_t = SliceAttentionFusion(feature_dim)
        self.fusion_m = SliceAttentionFusion(feature_dim)
        self.fusion_c = SliceAttentionFusion(feature_dim * 2)
        self.fusion_mt = SliceAttentionFusion(feature_dim)
        self.fusion_tm = SliceAttentionFusion(feature_dim)

        # 分类器
        self.lp_txt = nn.Linear(feature_dim, num_classes)
        self.lp_img = nn.Linear(feature_dim, num_classes)
        self.lp_cat = nn.Linear(feature_dim * 2, num_classes)
        self.lp_mt = nn.Linear(feature_dim, num_classes)
        self.lp_tm = nn.Linear(feature_dim, num_classes)

        self.meta_classifier = nn.Linear(5 * num_classes, num_classes)

    def forward(self, txt_feat, img_feat, mask):
        # 输入: (Batch, Slices, 512)
        B, S, D = txt_feat.shape

        # 1. 展平数据以进行 SADG 处理 (Batch * Slices, 512)
        flat_t = txt_feat.view(B * S, D)
        flat_m = img_feat.view(B * S, D)

        # 2. 归一化 (SADG 计算相似度需要归一化特征)
        flat_t = F.normalize(flat_t, dim=-1)
        flat_m = F.normalize(flat_m, dim=-1)

        # --- [修改点 2] 应用 SADG 门控进行去噪 ---
        # 这一步会抑制那些图文不符的切片特征
        gated_t, gated_m = self.sadg(flat_t, flat_m)

        # 3. 恢复形状 (Batch, Slices, 512) 用于后续融合
        feat_t = gated_t.view(B, S, D)
        feat_m = gated_m.view(B, S, D)

        # 4. 构建 5 视图 (此时使用的是去噪后的特征)
        feat_c = torch.cat((feat_t, feat_m), dim=-1)  # Concat

        # Cross Attention (注意：输入要是展平的)
        # 这里使用去噪后的 gated_t 和 gated_m
        flat_mt = self.cross_att_mt(gated_m, gated_t, gated_t)
        flat_tm = self.cross_att_tm(gated_t, gated_m, gated_m)

        feat_mt = flat_mt.view(B, S, D)
        feat_tm = flat_tm.view(B, S, D)

        # 5. 多切片融合 (聚合去噪后的切片)
        final_t = self.fusion_t(feat_t, mask)
        final_m = self.fusion_m(feat_m, mask)
        final_c = self.fusion_c(feat_c, mask)
        final_mt = self.fusion_mt(feat_mt, mask)
        final_tm = self.fusion_tm(feat_tm, mask)

        # 6. 分类
        logits_t = self.lp_txt(final_t)
        logits_m = self.lp_img(final_m)
        logits_c = self.lp_cat(final_c)
        logits_mt = self.lp_mt(final_mt)
        logits_tm = self.lp_tm(final_tm)

        meta_input = torch.cat((logits_t, logits_m, logits_c, logits_mt, logits_tm), dim=-1)
        final_logits = self.meta_classifier(meta_input)

        return final_logits