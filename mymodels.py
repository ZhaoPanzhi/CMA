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
    使用 TransformerEncoder 对 support 特征做 set-to-set 自适应，形成类原型；
    对 query 使用缩放 cosine 相似度进行度量分类。
    输入:
      - support_feats: [Ns, D]
      - support_labels: [Ns] (原始类标，非0..C-1也可)
      - query_feats: [Nq, D]
    输出:
      - logits: [Nq, C]  (C 为 support 中出现的类别数)
      - classes: [C]     (support 中的类别 ID 顺序，用于还原/对齐)
    """
    def __init__(self, in_dim=1024, num_heads=4, depth=1, proto_mlp=False, logit_scale=10.0):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=in_dim, nhead=num_heads, batch_first=True,
            dim_feedforward=in_dim*4, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.proto_proj = nn.Linear(in_dim, in_dim) if proto_mlp else nn.Identity()
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale)))  # 可学习缩放

    def forward(self, support_feats, support_labels, query_feats):
        # L2 norm
        s = F.normalize(support_feats, dim=-1)
        q = F.normalize(query_feats, dim=-1)

        # set-to-set 适配（仅对 support）
        s_adapt = self.encoder(s.unsqueeze(0)).squeeze(0)  # [Ns, D]
        s_adapt = self.proto_proj(s_adapt)

        # 生成类原型
        classes = torch.unique(support_labels)
        protos = []
        for c in classes:
            protos.append(s_adapt[support_labels == c].mean(dim=0))
        protos = torch.stack(protos, dim=0)  # [C, D]

        # 度量分类（cosine）
        q = F.normalize(q, dim=-1)
        protos = F.normalize(protos, dim=-1)
        logits = self.logit_scale * (q @ protos.t())  # [Nq, C]
        return logits, classes


