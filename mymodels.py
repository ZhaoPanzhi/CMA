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
        self.cross = nn.Linear(feature_dim, 2)

    def forward(self, txt_feat, img_feat):
        query = self.query(txt_feat)
        key = self.key(img_feat)
        value = self.value(img_feat)
        # print(key.size())

        attn_scores = torch.matmul(query, key.transpose(-2, -1))
        attn_scores = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_scores, value)
        attn_output = self.cross(attn_output)

        return attn_scores, attn_output


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)

        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class Adapter_V1(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(Adapter_V1, self).__init__()
        self.fc = nn.Linear(512 * 2, num_classes)
        self.fc_txt = nn.Linear(512, num_classes)
        self.fc_img = nn.Linear(512, num_classes)
        self.cross_attention = CrossAttention(512)
        self.fc_meta = nn.Linear(num_classes * 5, num_classes)

        # ========== 新增：模态 gating 模块 ==========
        # self.txt_gate = nn.Sequential(
        #     nn.Linear(512, 1),
        #     nn.Sigmoid()
        # )
        # self.img_gate = nn.Sequential(
        #     nn.Linear(512, 1),
        #     nn.Sigmoid()
        # )

        # ===== Competitive Modality Gating (CMG) =====
        # 输入是 text_feat + img_feat 拼接后的 1024 维特征，输出两个 gate：[alpha_t, alpha_i]
        self.gate_fc = nn.Linear(512 * 2, 2)  # 输出两个分数
        self.gate_softmax = nn.Softmax(dim=-1)

        # # ===== Cross-Modal Competitive Gating (CMG-X) =====
        # self.gate_mlp = nn.Sequential(
        #     nn.Linear(512 * 2, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 2)  # 输出两个 gate logits
        # )
        # self.gate_softmax = nn.Softmax(dim=-1)

    def forward(self, txt, img, fused):
        # # ========== 新增 gate 计算 ==========
        # g_t = self.txt_gate(txt)  # [B,1]
        # g_i = self.img_gate(img)  # [B,1]
        #
        # # ========== gated 输出 ==========
        # txt_out = g_t * self.fc_txt(txt)
        # img_out = g_i * self.fc_img(img)

        # ======== Competitive Modality Gating ========
        # 拼接原始特征 [B, 1024]
        g_input = torch.cat([txt, img], dim=-1)

        # 输出 [B, 2]
        gate_logits = self.gate_fc(g_input)

        # softmax 得到竞争权重 [B, 2]
        gates = self.gate_softmax(gate_logits)

        alpha_t = gates[:, 0].unsqueeze(-1)  # [B,1]
        alpha_i = gates[:, 1].unsqueeze(-1)  # [B,1]

        # 2. 获取用于 Deep Supervision 的原始 Logits (Raw Logits)
        # 这些是模型原本对该模态的判断，不受门控影响
        raw_txt_logits = self.fc_txt(txt)
        raw_img_logits = self.fc_img(img)

        # 3. 应用门控 (Gating)
        # 关键修改：我们在特征层面或 Logit 层面应用门控
        # 这里沿用您原本的逻辑：对 Logit 加权
        txt_out = alpha_t * raw_txt_logits
        img_out = alpha_i * raw_img_logits

        # 方案 A：直接对 输入特征 进行加权 (更彻底) -> 推荐
        txt_gated_feat = alpha_t * txt
        img_gated_feat = alpha_i * img
        fused_gated_feat = torch.cat([txt_gated_feat, img_gated_feat], dim=-1)
        fused_out = self.fc(fused_gated_feat)

        # fused_out = self.fc(fused)

        attn_ti, ti_attn_out = self.cross_attention(txt_gated_feat, img_gated_feat)
        attn_it, it_attn_out = self.cross_attention(img_gated_feat, txt_gated_feat)

        combined_out = torch.cat((txt_out, img_out, fused_out, it_attn_out, ti_attn_out), dim=1)

        meta_out = self.fc_meta(combined_out)

        return txt_out, img_out, meta_out, raw_txt_logits, raw_img_logits


class FEATHead(nn.Module):
    """
    FEAT + Prototype-MLP 版本：
    - 先用 TransformerEncoder 对 support 做 set-to-set 适配；
    - 再通过一个小型 MLP 对“类原型”所在空间做非线性变换（Prototype-MLP）；
    - 最后用缩放 cosine 相似度做度量分类。
    """
    def __init__(
        self,
        in_dim: int = 1024,
        num_heads: int = 4,
        depth: int = 1,
        proto_mlp: bool = False,
        proto_mlp_hidden: int = None,
        logit_scale: float = 10.0,
    ):
        super().__init__()

        # FEAT 的 set-to-set 适配（对 support）
        enc_layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=in_dim * 4,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

        # Prototype-MLP：对适配后的 support 特征再做一次 MLP 变换
        self.use_proto_mlp = proto_mlp
        if proto_mlp:
            if proto_mlp_hidden is None:
                proto_mlp_hidden = in_dim  # 也可以改成 2*in_dim
            self.proto_mlp = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, proto_mlp_hidden),
                nn.GELU(),
                nn.Linear(proto_mlp_hidden, in_dim),
            )
        else:
            self.proto_mlp = nn.Identity()

        # 可学习的 logit 缩放
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale)))

    def forward(self, support_feats, support_labels, query_feats):
        """
        support_feats: [Ns, D]
        support_labels: [Ns]
        query_feats:   [Nq, D]
        """
        # 归一化
        s = F.normalize(support_feats, dim=-1)
        q = F.normalize(query_feats, dim=-1)

        # 1) FEAT: set-to-set encoder 适配 support
        s_adapt = self.encoder(s.unsqueeze(0)).squeeze(0)  # [Ns, D]

        # 2) Prototype-MLP: 对 support 特征再做一次 MLP 变换
        s_adapt = self.proto_mlp(s_adapt)  # [Ns, D]

        # 3) 聚合成“类原型”
        classes = torch.unique(support_labels)
        protos = []
        for c in classes:
            protos.append(s_adapt[support_labels == c].mean(dim=0))
        protos = torch.stack(protos, dim=0)  # [C, D]

        # 4) cosine 度量分类
        q = F.normalize(q, dim=-1)
        protos = F.normalize(protos, dim=-1)
        logits = self.logit_scale * (q @ protos.t())  # [Nq, C]

        return logits, classes