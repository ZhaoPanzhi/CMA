import torch
import torch.nn as nn
import torch.nn.functional as F


class Adapter_Origin(nn.Module):
    def __init__(self, num_classes=2):
        super(Adapter_Origin, self).__init__()
        self.fc = nn.Linear(512 * 2, num_classes)

    def forward(self, x):
        return self.fc(x)


class SADG_Gating(nn.Module):
    """
    Semantic-aware dynamic gating for slice-level text/image features.

    Input:
        txt_feat, img_feat: [N, D]
    Output:
        gated_t, gated_m: [N, D]
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
        pt = self.proj_t(txt_feat)
        pm = self.proj_m(img_feat)
        conflict = torch.abs(pt - pm)

        gate_in = torch.cat([pt, pm, conflict], dim=-1).unsqueeze(-1)
        gate = torch.sigmoid(self.gate_conv(gate_in).squeeze(-1))

        gated_t = txt_feat * (1.0 + self.alpha * gate)
        gated_m = img_feat * (1.0 + self.alpha * gate)

        return gated_t, gated_m


class SliceAttentionFusion(nn.Module):
    """
    Attention fusion over valid long-image slices.

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
        scores = self.attn_fc(x).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        fused = torch.sum(x * weights, dim=1)
        return fused, weights.squeeze(-1)


class LivePrototypeAuxiliary(nn.Module):
    """
    Semantic prototype branch for high-level marketing risk patterns.

    This keeps the best-performing branch shape: a global K-prototype bank,
    conflict-guided prototype score refinement, and one auxiliary proto logits
    output consumed by the four-branch meta classifier.
    """
    def __init__(
        self,
        feature_dim=512,
        num_classes=2,
        num_prototypes=4,
        dropout=0.1,
        proto_temperature=1.0,
        use_conflict_input=True,
        use_conflict_bias=True
    ):
        super(LivePrototypeAuxiliary, self).__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.proto_temperature = proto_temperature
        self.use_conflict_input = use_conflict_input
        self.use_conflict_bias = use_conflict_bias

        self.live_proj = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.prototype_bank = nn.Parameter(torch.randn(num_prototypes, feature_dim) * 0.02)

        self.conflict_bias = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, num_prototypes)
        )

        self.proto_classifier = nn.Sequential(
            nn.Linear(feature_dim + num_prototypes, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, final_t, final_m):
        conflict = torch.abs(final_t - final_m)
        proto_conflict = conflict if self.use_conflict_input else torch.zeros_like(conflict)
        live_input = torch.cat([final_t, proto_conflict], dim=-1)
        z_live = self.live_proj(live_input)

        z_norm = F.normalize(z_live, dim=-1)
        p_norm = F.normalize(self.prototype_bank, dim=-1)

        proto_scores = torch.matmul(z_norm, p_norm.t())
        proto_scores = proto_scores / self.proto_temperature

        if self.use_conflict_bias:
            bias = self.conflict_bias(conflict)
        else:
            bias = torch.zeros_like(proto_scores)
        proto_scores_refined = proto_scores + bias

        proto_weights = F.softmax(proto_scores_refined, dim=-1)
        proto_repr = torch.matmul(proto_weights, self.prototype_bank)

        proto_feat = torch.cat([proto_repr, proto_scores_refined], dim=-1)
        logits_proto = self.proto_classifier(proto_feat)

        aux = {
            "z_live": z_live,
            "conflict": conflict,
            "proto_scores": proto_scores,
            "proto_scores_refined": proto_scores_refined,
            "proto_weights": proto_weights,
            "proto_class_scores": logits_proto,
            "proto_repr": proto_repr
        }

        return logits_proto, aux


class CMA_Model(nn.Module):
    """
    Main CMA model:
        1. SADG slice-level dynamic gating.
        2. Multi-slice attention fusion.
        3. Class-aware semantic prototype enhancement.

    Default forward returns final logits and stays compatible with existing
    training/evaluation code. Set return_aux=True to train/inspect prototype
    constraints.
    """
    def __init__(
        self,
        feature_dim=512,
        num_classes=2,
        num_prototypes=4,
        proto_temperature=1.0,
        use_sadg=True,
        use_slice_attention=True,
        use_proto=True,
        use_conflict_input=True,
        use_conflict_bias=True
    ):
        super(CMA_Model, self).__init__()

        self.sadg = SADG_Gating(feature_dim)
        self.use_sadg = use_sadg
        self.use_slice_attention = use_slice_attention
        self.use_proto = use_proto

        self.fusion_t = SliceAttentionFusion(feature_dim)
        self.fusion_m = SliceAttentionFusion(feature_dim)
        self.fusion_c = SliceAttentionFusion(feature_dim * 2)

        self.lp_txt = nn.Linear(feature_dim, num_classes)
        self.lp_img = nn.Linear(feature_dim, num_classes)
        self.lp_cat = nn.Linear(feature_dim * 2, num_classes)

        self.live_proto_aux = LivePrototypeAuxiliary(
            feature_dim=feature_dim,
            num_classes=num_classes,
            num_prototypes=num_prototypes,
            dropout=0.1,
            proto_temperature=proto_temperature,
            use_conflict_input=use_conflict_input,
            use_conflict_bias=use_conflict_bias
        )

        # Keep the best-performing output structure: four branch logits are
        # learned jointly by the meta classifier.
        self.meta_classifier = nn.Sequential(
            nn.Linear(num_classes * 4, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )

    def forward(self, txt_feat, img_feat, mask, return_aux=False, return_proto=False):
        B, S, D = txt_feat.shape

        flat_t = F.normalize(txt_feat.reshape(B * S, D), dim=-1)
        flat_m = F.normalize(img_feat.reshape(B * S, D), dim=-1)
        if self.use_sadg:
            gated_t, gated_m = self.sadg(flat_t, flat_m)
        else:
            gated_t, gated_m = flat_t, flat_m

        feat_t = gated_t.view(B, S, D)
        feat_m = gated_m.view(B, S, D)

        mask_u = mask.unsqueeze(-1).float()
        feat_t = feat_t * mask_u
        feat_m = feat_m * mask_u

        feat_c = torch.cat([feat_t, feat_m], dim=-1)

        if self.use_slice_attention:
            final_t, attn_t = self.fusion_t(feat_t, mask)
            final_m, attn_m = self.fusion_m(feat_m, mask)
            final_c, attn_c = self.fusion_c(feat_c, mask)
        else:
            final_t, attn_t = self.masked_mean(feat_t, mask)
            final_m, attn_m = self.masked_mean(feat_m, mask)
            final_c, attn_c = self.masked_mean(feat_c, mask)

        logits_t = self.lp_txt(final_t)
        logits_m = self.lp_img(final_m)
        logits_c = self.lp_cat(final_c)

        if self.use_proto:
            logits_proto, proto_aux = self.live_proto_aux(final_t, final_m)
        else:
            logits_proto = torch.zeros_like(logits_t)
            proto_aux = self.empty_proto_aux(final_t, final_m, logits_proto)

        meta_input = torch.cat([logits_t, logits_m, logits_c, logits_proto], dim=-1)
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

    @staticmethod
    def masked_mean(x, mask):
        weights = mask.float()
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        fused = torch.sum(x * weights.unsqueeze(-1), dim=1) / denom
        weights = weights / denom
        return fused, weights

    def empty_proto_aux(self, final_t, final_m, logits_proto):
        B = final_t.size(0)
        K = self.live_proto_aux.num_prototypes
        zeros_scores = final_t.new_zeros(B, K)
        zeros_repr = final_t.new_zeros(B, final_t.size(-1))
        return {
            "z_live": zeros_repr,
            "conflict": torch.abs(final_t - final_m),
            "proto_scores": zeros_scores,
            "proto_scores_refined": zeros_scores,
            "proto_weights": zeros_scores,
            "proto_class_scores": logits_proto,
            "proto_repr": zeros_repr
        }

    def prototype_diversity_loss(self):
        prototypes = F.normalize(self.live_proto_aux.prototype_bank, dim=-1)
        flat = prototypes.reshape(-1, prototypes.size(-1))
        sim = torch.matmul(flat, flat.t())
        eye = torch.eye(sim.size(0), device=sim.device, dtype=torch.bool)
        off_diag = sim.masked_select(~eye)
        return off_diag.pow(2).mean()
