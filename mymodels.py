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

        # # ===== Competitive Modality Gating (CMG) =====
        # # 输入是 text_feat + img_feat 拼接后的 1024 维特征，输出两个 gate：[alpha_t, alpha_i]
        # self.gate_fc = nn.Linear(512 * 2, 2)  # 输出两个分数
        # self.gate_softmax = nn.Softmax(dim=-1)

        #  通道级门控网络 (Channel-wise Gating Network)
        self.gate_net = nn.Sequential(
            nn.Linear(512 * 2, 256),  # 降维 (Squeeze): 压缩信息，减少参数量
            nn.ReLU(),  # 激活
            nn.Linear(256, 512 * 2),  # 升维 (Excitation): 恢复到通道维度
            nn.Sigmoid()  # 归一化: 将权重限制在 0~1 之间
        )

        # nn.init.constant_(self.gate_net[2].bias, 3.0)

    def forward(self, txt, img, fused):
        # # ========== 新增 gate 计算 ==========
        # g_t = self.txt_gate(txt)  # [B,1]
        # g_i = self.img_gate(img)  # [B,1]
        #
        # # ========== gated 输出 ==========
        # txt_out = g_t * self.fc_txt(txt)
        # img_out = g_i * self.fc_img(img)

        # # ======== Competitive Modality Gating ========
        # # 拼接原始特征 [B, 1024]
        # g_input = torch.cat([txt, img], dim=-1)
        #
        # # 输出 [B, 2]
        # gate_logits = self.gate_fc(g_input)
        #
        # # softmax 得到竞争权重 [B, 2]
        # gates = self.gate_softmax(gate_logits)
        #
        # alpha_t = gates[:, 0].unsqueeze(-1)  # [B,1]
        # alpha_i = gates[:, 1].unsqueeze(-1)  # [B,1]
        #
        # # 2. 获取用于 Deep Supervision 的原始 Logits (Raw Logits)
        # # 这些是模型原本对该模态的判断，不受门控影响
        # raw_txt_logits = self.fc_txt(txt)
        # raw_img_logits = self.fc_img(img)
        #
        # # 3. 应用门控 (Gating)
        # # 关键修改：我们在特征层面或 Logit 层面应用门控
        # # 这里沿用您原本的逻辑：对 Logit 加权
        # txt_out = alpha_t * raw_txt_logits
        # img_out = alpha_i * raw_img_logits
        #
        # # 方案 A：直接对 输入特征 进行加权 (更彻底) -> 推荐
        # txt_gated_feat = alpha_t * txt
        # img_gated_feat = alpha_i * img
        # fused_gated_feat = torch.cat([txt_gated_feat, img_gated_feat], dim=-1)
        # fused_out = self.fc(fused_gated_feat)
        #
        # # fused_out = self.fc(fused)
        #
        # attn_ti, ti_attn_out = self.cross_attention(txt_gated_feat, img_gated_feat)
        # attn_it, it_attn_out = self.cross_attention(img_gated_feat, txt_gated_feat)
        #
        # combined_out = torch.cat((txt_out, img_out, fused_out, it_attn_out, ti_attn_out), dim=1)
        #
        # meta_out = self.fc_meta(combined_out)

        # 1. 拼接原始特征
        g_input = torch.cat([txt, img], dim=-1)  # [B, 1024]

        # 2. 通过门控网络生成权重向量
        gate_weights = self.gate_net(g_input)  # [B, 1024]

        # 3. 拆分权重给 Text 和 Image
        alpha_t = gate_weights[:, :512]  # [B, 512] - 对应文本每一维的权重
        alpha_i = gate_weights[:, 512:]  # [B, 512] - 对应图像每一维的权重

        # ============================================================
        # 【修改重点 3】: 应用门控 (Element-wise Product) & 准备 Deep Sup
        # ============================================================
        # 1. 保留原始 Logits 用于 Deep Supervision (不受门控影响)
        raw_txt_logits = self.fc_txt(F.normalize(txt, dim=-1))
        raw_img_logits = self.fc_img(F.normalize(img, dim=-1))

        # 2. 生成“干净”的特征 (Gated Features)
        # 逐元素相乘：噪声维度的权重会被网络自动压低
        txt_gated_feat = txt * alpha_t
        img_gated_feat = img * alpha_i

        # ⚠️ 非常重要：重新归一化！防止特征变小导致 Attention 失效
        txt_gated_feat = F.normalize(txt_gated_feat, dim=-1)
        img_gated_feat = F.normalize(img_gated_feat, dim=-1)

        # 3. 基于干净特征计算输出 (用于最终融合)
        txt_out = self.fc_txt(txt_gated_feat)
        img_out = self.fc_img(img_gated_feat)

        # ============================================================
        # 【修改重点 4】: 全链路使用干净特征 (堵住漏洞)
        # ============================================================

        # A. Fused 分支：重新拼接干净特征
        fused_gated_feat = torch.cat([txt_gated_feat, img_gated_feat], dim=-1)
        fused_out = self.fc(fused_gated_feat)

        # B. Cross-Attention 分支：⚠️ 必须使用 Gated Feat
        # 修复了之前 32-shot 下降的问题，防止噪声通过 Attention 泄漏
        attn_ti, ti_attn_out = self.cross_attention(txt_gated_feat, img_gated_feat)
        attn_it, it_attn_out = self.cross_attention(img_gated_feat, txt_gated_feat)

        # ============================================================
        # 最终融合
        # ============================================================
        combined_out = torch.cat((txt_out, img_out, fused_out, it_attn_out, ti_attn_out), dim=1)
        meta_out = self.fc_meta(combined_out)

        return txt_out, img_out, meta_out, raw_txt_logits, raw_img_logits
