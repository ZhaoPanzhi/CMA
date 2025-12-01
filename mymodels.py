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

    def forward(self, txt, img, fused):
        txt_out = self.fc_txt(txt)
        img_out = self.fc_img(img)
        fused_out = self.fc(fused)

        attn_ti, ti_attn_out = self.cross_attention(txt, img)
        attn_it, it_attn_out = self.cross_attention(img, txt)

        combined_out = torch.cat((txt_out, img_out, fused_out, it_attn_out, ti_attn_out), dim=1)

        meta_out = self.fc_meta(combined_out)

        return txt_out, img_out, meta_out


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