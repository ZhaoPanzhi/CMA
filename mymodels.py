import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter_Origin(nn.Module):
    def __init__(self, num_classes=2):
        super(Adapter_Origin, self).__init__()
        self.fc = nn.Linear(512 * 2, num_classes)

    def forward(self, x):
        return self.fc(x)


# =========================================================
# 1. SADG: 保留第一部分核心模块
# =========================================================
class SADG_Gating(nn.Module):
    """
    输入: txt_feat, img_feat -> [N, D]
    输出: gated_t, gated_m -> [N, D]
    """
    def __init__(self, feature_dim=512):
        super(SADG_Gating, self).__init__()

        self.proj_t = nn.Linear(feature_dim, feature_dim)
        self.proj_m = nn.Linear(feature_dim, feature_dim)

        self.gate_conv = nn.Conv1d(
            in_channels=feature_dim * 3,
            out_channels=feature_dim,
            kernel_size=1,
            bias=True
        )

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, txt_feat, img_feat):
        pt = self.proj_t(txt_feat)   # [N, D]
        pm = self.proj_m(img_feat)   # [N, D]

        conflict = torch.abs(pt - pm)  # [N, D]

        gate_in = torch.cat([pt, pm, conflict], dim=-1)  # [N, 3D]
        gate_in = gate_in.unsqueeze(-1)                  # [N, 3D, 1]
        gate = torch.sigmoid(self.gate_conv(gate_in).squeeze(-1))  # [N, D]

        gated_t = txt_feat * (1.0 + self.alpha * gate)
        gated_m = img_feat * (1.0 + self.alpha * gate)

        return gated_t, gated_m


# =========================================================
# 2. 多切片融合：保留
# =========================================================
class SliceAttentionFusion(nn.Module):
    """
    x:    [B, S, D]
    mask: [B, S]
    """
    def __init__(self, feature_dim=512, hidden_dim=128):
        super(SliceAttentionFusion, self).__init__()
        self.attn_fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, mask):
        scores = self.attn_fc(x).squeeze(-1)  # [B, S]
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # [B, S, 1]
        fused = torch.sum(x * weights, dim=1)  # [B, D]
        return fused, weights.squeeze(-1)


# =========================================================
# 3. 直播营销语义原型辅助分支（保留原最佳版本）
# =========================================================
class LivePrototypeAuxiliary(nn.Module):
    """
    输入:
        final_t: [B, D]
        final_m: [B, D]

    思路:
        用 text 表达 + conflict 表达 构建直播营销语义表示，
        再与 prototype bank 做匹配，
        输出一个辅助 logits_proto
    """
    def __init__(self, feature_dim=512, num_classes=2, num_prototypes=4, dropout=0.1):
        super(LivePrototypeAuxiliary, self).__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes

        # 更贴近直播营销：文本 + 冲突
        self.live_proj = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 可学习直播营销语义原型
        self.prototype_bank = nn.Parameter(
            torch.randn(num_prototypes, feature_dim) * 0.02
        )

        # conflict 引导的 prototype 修正
        self.conflict_bias = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, num_prototypes)
        )

        # 原型表示 -> 辅助分类
        self.proto_classifier = nn.Sequential(
            nn.Linear(feature_dim + num_prototypes, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, final_t, final_m):
        """
        final_t: [B, D]
        final_m: [B, D]
        """
        conflict = torch.abs(final_t - final_m)  # [B, D]

        # 直播营销表示：text + conflict
        live_input = torch.cat([final_t, conflict], dim=-1)  # [B, 2D]
        z_live = self.live_proj(live_input)  # [B, D]

        # prototype matching
        z_norm = F.normalize(z_live, dim=-1)                # [B, D]
        p_norm = F.normalize(self.prototype_bank, dim=-1)   # [K, D]

        proto_scores = torch.matmul(z_norm, p_norm.t())     # [B, K]
        bias = self.conflict_bias(conflict)                 # [B, K]
        proto_scores_refined = proto_scores + bias          # [B, K]

        proto_weights = F.softmax(proto_scores_refined, dim=-1)  # [B, K]
        proto_repr = torch.matmul(proto_weights, self.prototype_bank)  # [B, D]

        # 输出辅助 logits
        proto_feat = torch.cat([proto_repr, proto_scores_refined], dim=-1)  # [B, D+K]
        logits_proto = self.proto_classifier(proto_feat)  # [B, C]

        aux = {
            "z_live": z_live,
            "conflict": conflict,
            "proto_scores": proto_scores,
            "proto_scores_refined": proto_scores_refined,
            "proto_weights": proto_weights,
            "proto_repr": proto_repr
        }

        return logits_proto, aux


# =========================================================
# 4. 主模型：3-view 主分类 + live prototype 辅助分支
#    仅增加 return_proto 接口，不改变原始结构和计算逻辑
# =========================================================
class CMA_Model(nn.Module):
    """
    默认 forward 只返回 logits，兼容原训练脚本
    可选:
        return_aux=True   -> 返回 final_logits, aux
        return_proto=True -> 返回 final_logits, logits_proto
    注意:
        该版本仅做最小兼容修改，不改变原最佳版本计算图
    """
    def __init__(self, feature_dim=512, num_classes=2, num_prototypes=4):
        super(CMA_Model, self).__init__()

        # -------- 第一部分保留 --------
        self.sadg = SADG_Gating(feature_dim)

        self.fusion_t = SliceAttentionFusion(feature_dim)
        self.fusion_m = SliceAttentionFusion(feature_dim)
        self.fusion_c = SliceAttentionFusion(feature_dim * 2)

        # -------- 3-view 主分类头 --------
        self.lp_txt = nn.Linear(feature_dim, num_classes)
        self.lp_img = nn.Linear(feature_dim, num_classes)
        self.lp_cat = nn.Linear(feature_dim * 2, num_classes)

        # -------- 直播营销语义辅助分支 --------
        self.live_proto_aux = LivePrototypeAuxiliary(
            feature_dim=feature_dim,
            num_classes=num_classes,
            num_prototypes=num_prototypes,
            dropout=0.1
        )

        # -------- 最终融合分类 --------
        # [logits_t, logits_m, logits_c, logits_proto] -> 4C
        self.meta_classifier = nn.Sequential(
            nn.Linear(num_classes * 4, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )

    def forward(self, txt_feat, img_feat, mask, return_aux=False, return_proto=False):
        """
        txt_feat: [B, S, D]
        img_feat: [B, S, D]
        mask:     [B, S]
        """
        B, S, D = txt_feat.shape

        # 1) flatten -> SADG
        flat_t = txt_feat.reshape(B * S, D)
        flat_m = img_feat.reshape(B * S, D)

        flat_t = F.normalize(flat_t, dim=-1)
        flat_m = F.normalize(flat_m, dim=-1)

        gated_t, gated_m = self.sadg(flat_t, flat_m)

        # 2) reshape back
        feat_t = gated_t.view(B, S, D)
        feat_m = gated_m.view(B, S, D)

        # 避免 padding 污染
        mask_u = mask.unsqueeze(-1).float()  # [B, S, 1]
        feat_t = feat_t * mask_u
        feat_m = feat_m * mask_u

        # 3) concat view
        feat_c = torch.cat([feat_t, feat_m], dim=-1)  # [B, S, 2D]

        # 4) 多切片融合
        final_t, attn_t = self.fusion_t(feat_t, mask)  # [B, D]
        final_m, attn_m = self.fusion_m(feat_m, mask)  # [B, D]
        final_c, attn_c = self.fusion_c(feat_c, mask)  # [B, 2D]

        # 5) 主分类分支
        logits_t = self.lp_txt(final_t)    # [B, C]
        logits_m = self.lp_img(final_m)    # [B, C]
        logits_c = self.lp_cat(final_c)    # [B, C]

        # 6) 直播营销语义辅助分支
        logits_proto, proto_aux = self.live_proto_aux(final_t, final_m)  # [B, C]

        # 7) 最终融合
        meta_input = torch.cat([logits_t, logits_m, logits_c, logits_proto], dim=-1)  # [B, 4C]
        final_logits = self.meta_classifier(meta_input)

        if return_aux:
            aux = {
                "slice_attn_t": attn_t,
                "slice_attn_m": attn_m,
                "slice_attn_c": attn_c,
                "logits_t": logits_t,
                "logits_m": logits_m,
                "logits_c": logits_c,
                "logits_proto": logits_proto,
                **proto_aux
            }
            return final_logits, aux

        if return_proto:
            return final_logits, logits_proto

        return final_logits