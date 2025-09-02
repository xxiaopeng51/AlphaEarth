"""
Loss Functions

多种损失函数的实现，支持多模态对比学习、重建和一致性约束。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import math
from einops import rearrange


class MultiModalContrastiveLoss(nn.Module):
    """
    多模态对比学习损失
    
    扩展CLIP的对比学习到多种模态组合：
    - 图像-文本对比
    - 跨时间对比
    - 跨模态对比
    - 跨区域对比
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        modality_weights: Optional[Dict[str, float]] = None,
        use_hard_negatives: bool = True,
        hard_negative_ratio: float = 0.1
    ):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.modality_weights = modality_weights or {}
        self.use_hard_negatives = use_hard_negatives
        self.hard_negative_ratio = hard_negative_ratio
    
    def forward(
        self,
        image_embeddings: torch.Tensor,      # (B, H, W, D) 或 (B, D)
        text_embeddings: Optional[torch.Tensor] = None,  # (B, D)
        temporal_embeddings: Optional[torch.Tensor] = None,  # (B, T, D)
        modality_embeddings: Optional[Dict[str, torch.Tensor]] = None,  # {modality: (B, D)}
        labels: Optional[torch.Tensor] = None  # (B,) 样本标签
    ) -> Dict[str, torch.Tensor]:
        """
        计算多模态对比损失
        
        Args:
            image_embeddings: 图像嵌入
            text_embeddings: 文本嵌入
            temporal_embeddings: 时间序列嵌入
            modality_embeddings: 各模态嵌入
            labels: 样本标签 (用于hard negative mining)
        """
        losses = {}
        total_loss = 0.0
        
        # 1. 图像-文本对比损失
        if text_embeddings is not None:
            # 如果图像嵌入是空间的，需要池化
            if image_embeddings.dim() == 4:  # (B, H, W, D)
                img_emb = image_embeddings.mean(dim=(1, 2))  # (B, D)
            else:
                img_emb = image_embeddings
            
            image_text_loss = self._compute_contrastive_loss(
                img_emb, text_embeddings, "image_text"
            )
            losses["image_text_loss"] = image_text_loss
            total_loss += image_text_loss * self.modality_weights.get("image_text", 1.0)
        
        # 2. 时间对比损失
        if temporal_embeddings is not None:
            temporal_loss = self._compute_temporal_contrastive_loss(temporal_embeddings)
            losses["temporal_loss"] = temporal_loss
            total_loss += temporal_loss * self.modality_weights.get("temporal", 0.5)
        
        # 3. 跨模态对比损失
        if modality_embeddings is not None and len(modality_embeddings) > 1:
            cross_modal_loss = self._compute_cross_modal_loss(modality_embeddings)
            losses["cross_modal_loss"] = cross_modal_loss
            total_loss += cross_modal_loss * self.modality_weights.get("cross_modal", 0.3)
        
        # 4. Hard negative mining (如果启用)
        if self.use_hard_negatives and labels is not None:
            hard_neg_loss = self._compute_hard_negative_loss(image_embeddings, labels)
            losses["hard_negative_loss"] = hard_neg_loss
            total_loss += hard_neg_loss * self.modality_weights.get("hard_negative", 0.2)
        
        losses["total_contrastive_loss"] = total_loss
        return losses
    
    def _compute_contrastive_loss(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
        loss_name: str
    ) -> torch.Tensor:
        """计算两个嵌入之间的对比损失"""
        # L2标准化
        a_norm = F.normalize(embeddings_a, p=2, dim=-1)
        b_norm = F.normalize(embeddings_b, p=2, dim=-1)
        
        # 计算相似度矩阵
        similarity = torch.matmul(a_norm, b_norm.T) / self.temperature  # (B, B)
        
        # 标签 (对角线为正样本)
        batch_size = similarity.shape[0]
        labels = torch.arange(batch_size, device=similarity.device)
        
        # 对称的对比损失
        loss_a_to_b = F.cross_entropy(similarity, labels)
        loss_b_to_a = F.cross_entropy(similarity.T, labels)
        
        return (loss_a_to_b + loss_b_to_a) / 2
    
    def _compute_temporal_contrastive_loss(self, temporal_embeddings: torch.Tensor) -> torch.Tensor:
        """计算时间对比损失"""
        B, T, D = temporal_embeddings.shape
        
        # 选择锚点时间步和正负样本
        anchor_indices = torch.randint(0, T, (B,), device=temporal_embeddings.device)
        positive_indices = torch.clamp(anchor_indices + torch.randint(-2, 3, (B,)), 0, T-1)
        
        anchors = temporal_embeddings[torch.arange(B), anchor_indices]  # (B, D)
        positives = temporal_embeddings[torch.arange(B), positive_indices]  # (B, D)
        
        # 负样本：其他样本的随机时间步
        negative_indices = torch.randint(0, T, (B,))
        negatives = temporal_embeddings[torch.randperm(B), negative_indices]  # (B, D)
        
        # 计算对比损失
        anchor_norm = F.normalize(anchors, p=2, dim=-1)
        positive_norm = F.normalize(positives, p=2, dim=-1)
        negative_norm = F.normalize(negatives, p=2, dim=-1)
        
        pos_sim = torch.sum(anchor_norm * positive_norm, dim=-1) / self.temperature  # (B,)
        neg_sim = torch.sum(anchor_norm * negative_norm, dim=-1) / self.temperature  # (B,)
        
        # InfoNCE损失
        logits = torch.stack([pos_sim, neg_sim], dim=-1)  # (B, 2)
        labels = torch.zeros(B, dtype=torch.long, device=logits.device)
        
        return F.cross_entropy(logits, labels)
    
    def _compute_cross_modal_loss(self, modality_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算跨模态对比损失"""
        modalities = list(modality_embeddings.keys())
        total_loss = 0.0
        num_pairs = 0
        
        # 计算所有模态对之间的对比损失
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                mod_a, mod_b = modalities[i], modalities[j]
                emb_a = modality_embeddings[mod_a]
                emb_b = modality_embeddings[mod_b]
                
                # 如果嵌入是空间的，需要池化
                if emb_a.dim() > 2:
                    emb_a = emb_a.mean(dim=tuple(range(1, emb_a.dim()-1)))
                if emb_b.dim() > 2:
                    emb_b = emb_b.mean(dim=tuple(range(1, emb_b.dim()-1)))
                
                pair_loss = self._compute_contrastive_loss(emb_a, emb_b, f"{mod_a}_{mod_b}")
                total_loss += pair_loss
                num_pairs += 1
        
        return total_loss / max(num_pairs, 1)
    
    def _compute_hard_negative_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """计算hard negative挖掘损失"""
        if embeddings.dim() > 2:
            embeddings = embeddings.mean(dim=tuple(range(1, embeddings.dim()-1)))
        
        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        similarity = torch.matmul(embeddings_norm, embeddings_norm.T)
        
        # 找到hard negatives (同类中相似度最低的，异类中相似度最高的)
        batch_size = embeddings.shape[0]
        mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        
        # 同类样本中的hard negatives
        same_class_sim = similarity.masked_fill(~mask, float('-inf'))
        hard_negatives_same = same_class_sim.topk(k=max(1, int(batch_size * self.hard_negative_ratio)), 
                                                 dim=-1, largest=False).values
        
        # 异类样本中的hard negatives
        diff_class_sim = similarity.masked_fill(mask, float('-inf'))
        hard_negatives_diff = diff_class_sim.topk(k=max(1, int(batch_size * self.hard_negative_ratio)), 
                                                 dim=-1, largest=True).values
        
        # 构建hard negative损失
        pos_sim = torch.diagonal(similarity)  # 自己和自己的相似度作为正样本
        
        # 使用hard negatives计算损失
        neg_sim = torch.cat([hard_negatives_same, hard_negatives_diff], dim=-1)
        
        # InfoNCE损失
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1) / self.temperature
        labels_hn = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        
        return F.cross_entropy(logits, labels_hn)


class ReconstructionLoss(nn.Module):
    """
    重建损失
    
    支持多种重建任务和损失函数组合
    """
    
    def __init__(
        self,
        loss_types: List[str] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        perceptual_weight: float = 0.1
    ):
        super().__init__()
        
        if loss_types is None:
            loss_types = ["l1", "l2", "ssim", "perceptual"]
        
        self.loss_types = loss_types
        self.loss_weights = loss_weights or {loss_type: 1.0 for loss_type in loss_types}
        self.perceptual_weight = perceptual_weight
        
        # 感知损失网络 (使用预训练的特征提取器)
        if "perceptual" in loss_types:
            self.perceptual_net = self._create_perceptual_network()
    
    def _create_perceptual_network(self) -> nn.Module:
        """创建感知损失网络"""
        # 使用预训练的ResNet特征
        import timm
        backbone = timm.create_model('resnet50', pretrained=True, features_only=True)
        
        # 冻结参数
        for param in backbone.parameters():
            param.requires_grad = False
        
        return backbone
    
    def forward(
        self,
        reconstructed: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        modality_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算重建损失
        
        Args:
            reconstructed: 重建结果 {modality: (B, C, H, W)}
            target: 目标数据 {modality: (B, C, H, W)}
            modality_weights: 模态权重
        """
        losses = {}
        total_loss = 0.0
        
        if modality_weights is None:
            modality_weights = {mod: 1.0 for mod in reconstructed.keys()}
        
        for modality in reconstructed.keys():
            if modality not in target:
                continue
            
            recon = reconstructed[modality]
            tgt = target[modality]
            
            modality_loss = 0.0
            
            # L1损失
            if "l1" in self.loss_types:
                l1_loss = F.l1_loss(recon, tgt)
                losses[f"{modality}_l1"] = l1_loss
                modality_loss += l1_loss * self.loss_weights.get("l1", 1.0)
            
            # L2损失
            if "l2" in self.loss_types:
                l2_loss = F.mse_loss(recon, tgt)
                losses[f"{modality}_l2"] = l2_loss
                modality_loss += l2_loss * self.loss_weights.get("l2", 1.0)
            
            # SSIM损失
            if "ssim" in self.loss_types:
                ssim_loss = 1 - self._compute_ssim(recon, tgt)
                losses[f"{modality}_ssim"] = ssim_loss
                modality_loss += ssim_loss * self.loss_weights.get("ssim", 1.0)
            
            # 感知损失
            if "perceptual" in self.loss_types and hasattr(self, 'perceptual_net'):
                perceptual_loss = self._compute_perceptual_loss(recon, tgt)
                losses[f"{modality}_perceptual"] = perceptual_loss
                modality_loss += perceptual_loss * self.loss_weights.get("perceptual", 1.0)
            
            losses[f"{modality}_total"] = modality_loss
            total_loss += modality_loss * modality_weights.get(modality, 1.0)
        
        losses["total_reconstruction_loss"] = total_loss
        return losses
    
    def _compute_ssim(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        window_size: int = 11,
        size_average: bool = True
    ) -> torch.Tensor:
        """计算SSIM"""
        def gaussian_window(size: int, sigma: float = 1.5) -> torch.Tensor:
            coords = torch.arange(size, dtype=torch.float32)
            coords -= size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g /= g.sum()
            return g.unsqueeze(0) * g.unsqueeze(1)
        
        window = gaussian_window(window_size).to(x.device)
        window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # 计算局部均值
        mu1 = F.conv2d(x, window, padding=window_size//2, groups=x.shape[1])
        mu2 = F.conv2d(y, window, padding=window_size//2, groups=y.shape[1])
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # 计算局部方差和协方差
        sigma1_sq = F.conv2d(x * x, window, padding=window_size//2, groups=x.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(y * y, window, padding=window_size//2, groups=y.shape[1]) - mu2_sq
        sigma12 = F.conv2d(x * y, window, padding=window_size//2, groups=x.shape[1]) - mu1_mu2
        
        # SSIM计算
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(dim=(1, 2, 3))
    
    def _compute_perceptual_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算感知损失"""
        if not hasattr(self, 'perceptual_net'):
            return torch.tensor(0.0, device=x.device)
        
        # 确保输入是3通道 (RGB)
        if x.shape[1] != 3:
            # 如果不是3通道，选择前3个通道或重复
            if x.shape[1] >= 3:
                x = x[:, :3]
                y = y[:, :3]
            else:
                x = x.repeat(1, 3//x.shape[1] + 1, 1, 1)[:, :3]
                y = y.repeat(1, 3//y.shape[1] + 1, 1, 1)[:, :3]
        
        # 提取特征
        with torch.no_grad():
            features_x = self.perceptual_net(x)
            features_y = self.perceptual_net(y)
        
        # 计算特征距离
        perceptual_loss = 0.0
        for feat_x, feat_y in zip(features_x, features_y):
            perceptual_loss += F.mse_loss(feat_x, feat_y)
        
        return perceptual_loss


class ConsistencyLoss(nn.Module):
    """
    一致性损失
    
    确保Teacher-Student网络和不同视图之间的一致性
    """
    
    def __init__(
        self,
        consistency_types: List[str] = None,
        temperature: float = 0.1,
        alpha: float = 0.99  # EMA更新系数
    ):
        super().__init__()
        
        if consistency_types is None:
            consistency_types = ["teacher_student", "temporal", "spatial", "cross_modal"]
        
        self.consistency_types = consistency_types
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(
        self,
        teacher_outputs: Dict[str, torch.Tensor],
        student_outputs: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """计算一致性损失"""
        losses = {}
        total_loss = 0.0
        
        # Teacher-Student一致性
        if "teacher_student" in self.consistency_types:
            ts_loss = self._compute_teacher_student_loss(teacher_outputs, student_outputs)
            losses["teacher_student_loss"] = ts_loss
            total_loss += ts_loss
        
        # 时间一致性
        if "temporal" in self.consistency_types and "temporal_features" in teacher_outputs:
            temporal_loss = self._compute_temporal_consistency_loss(
                teacher_outputs["temporal_features"]
            )
            losses["temporal_consistency_loss"] = temporal_loss
            total_loss += temporal_loss * 0.5
        
        # 空间一致性
        if "spatial" in self.consistency_types:
            spatial_loss = self._compute_spatial_consistency_loss(teacher_outputs, student_outputs)
            losses["spatial_consistency_loss"] = spatial_loss
            total_loss += spatial_loss * 0.3
        
        losses["total_consistency_loss"] = total_loss
        return losses
    
    def _compute_teacher_student_loss(
        self,
        teacher_outputs: Dict[str, torch.Tensor],
        student_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Teacher-Student一致性损失"""
        teacher_emb = teacher_outputs.get("embeddings")
        student_emb = student_outputs.get("embeddings")
        
        if teacher_emb is None or student_emb is None:
            return torch.tensor(0.0, device=next(iter(teacher_outputs.values())).device)
        
        # KL散度损失
        teacher_norm = F.normalize(teacher_emb, p=2, dim=-1)
        student_norm = F.normalize(student_emb, p=2, dim=-1)
        
        # 软目标
        teacher_soft = F.softmax(teacher_norm / self.temperature, dim=-1)
        student_log_soft = F.log_softmax(student_norm / self.temperature, dim=-1)
        
        kl_loss = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean')
        
        return kl_loss
    
    def _compute_temporal_consistency_loss(self, temporal_features: torch.Tensor) -> torch.Tensor:
        """时间一致性损失"""
        B, T, *spatial_dims, D = temporal_features.shape
        
        # 相邻时间步的特征应该相似
        if T > 1:
            current_features = temporal_features[:, :-1]  # (B, T-1, ...)
            next_features = temporal_features[:, 1:]      # (B, T-1, ...)
            
            # 计算相邻时间步的相似度
            current_flat = current_features.flatten(start_dim=2)  # (B, T-1, *)
            next_flat = next_features.flatten(start_dim=2)        # (B, T-1, *)
            
            # 余弦相似度
            current_norm = F.normalize(current_flat, p=2, dim=-1)
            next_norm = F.normalize(next_flat, p=2, dim=-1)
            
            similarity = torch.sum(current_norm * next_norm, dim=-1)  # (B, T-1)
            
            # 最大化相邻时间步的相似度
            temporal_loss = 1 - similarity.mean()
            
            return temporal_loss
        
        return torch.tensor(0.0, device=temporal_features.device)
    
    def _compute_spatial_consistency_loss(
        self,
        teacher_outputs: Dict[str, torch.Tensor],
        student_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """空间一致性损失"""
        teacher_emb = teacher_outputs.get("embeddings")
        student_emb = student_outputs.get("embeddings")
        
        if teacher_emb is None or student_emb is None:
            return torch.tensor(0.0, device=next(iter(teacher_outputs.values())).device)
        
        if teacher_emb.dim() != 4:  # 需要空间维度
            return torch.tensor(0.0, device=teacher_emb.device)
        
        B, H, W, D = teacher_emb.shape
        
        # 计算空间邻域的一致性
        # 上下左右邻域
        teacher_shifted = []
        student_shifted = []
        
        shifts = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 右、左、下、上
        
        for dy, dx in shifts:
            if dy == 0 and dx == 1:  # 右移
                t_shift = teacher_emb[:, :, 1:, :]
                s_shift = student_emb[:, :, 1:, :]
                t_orig = teacher_emb[:, :, :-1, :]
                s_orig = student_emb[:, :, :-1, :]
            elif dy == 0 and dx == -1:  # 左移
                t_shift = teacher_emb[:, :, :-1, :]
                s_shift = student_emb[:, :, :-1, :]
                t_orig = teacher_emb[:, :, 1:, :]
                s_orig = student_emb[:, :, 1:, :]
            elif dy == 1 and dx == 0:  # 下移
                t_shift = teacher_emb[:, 1:, :, :]
                s_shift = student_emb[:, 1:, :, :]
                t_orig = teacher_emb[:, :-1, :, :]
                s_orig = student_emb[:, :-1, :, :]
            elif dy == -1 and dx == 0:  # 上移
                t_shift = teacher_emb[:, :-1, :, :]
                s_shift = student_emb[:, :-1, :, :]
                t_orig = teacher_emb[:, 1:, :, :]
                s_orig = student_emb[:, 1:, :, :]
            
            teacher_shifted.append((t_orig, t_shift))
            student_shifted.append((s_orig, s_shift))
        
        # 计算邻域一致性
        spatial_loss = 0.0
        for (t_orig, t_shift), (s_orig, s_shift) in zip(teacher_shifted, student_shifted):
            # Teacher的空间一致性
            t_consistency = F.cosine_similarity(
                t_orig.flatten(start_dim=1), 
                t_shift.flatten(start_dim=1), 
                dim=-1
            ).mean()
            
            # Student的空间一致性
            s_consistency = F.cosine_similarity(
                s_orig.flatten(start_dim=1),
                s_shift.flatten(start_dim=1),
                dim=-1
            ).mean()
            
            # 一致性差异
            spatial_loss += F.mse_loss(t_consistency, s_consistency)
        
        return spatial_loss / len(shifts)


class VonMisesFisherLoss(nn.Module):
    """von Mises-Fisher分布损失"""
    
    def __init__(self, dim: int, kappa: float = 10.0):
        super().__init__()
        self.dim = dim
        self.kappa = nn.Parameter(torch.tensor(kappa))
    
    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算vMF损失
        
        Args:
            embeddings: (B, ..., dim) 预测嵌入 (单位向量)
            targets: (B, ..., dim) 目标嵌入 (单位向量)
        """
        # 确保是单位向量
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        targets = F.normalize(targets, p=2, dim=-1)
        
        # vMF负对数似然
        dot_product = torch.sum(embeddings * targets, dim=-1)
        
        # 近似的归一化常数
        log_norm_const = (self.dim / 2 - 1) * torch.log(self.kappa) - \
                        (self.dim / 2) * math.log(2 * math.pi)
        
        # 负对数似然
        nll = -log_norm_const - self.kappa * dot_product
        
        return nll.mean()


class CombinedLoss(nn.Module):
    """
    组合损失函数
    
    整合对比学习、重建和一致性损失
    """
    
    def __init__(
        self,
        loss_weights: Dict[str, float] = None,
        adaptive_weighting: bool = True
    ):
        super().__init__()
        
        default_weights = {
            "contrastive": 1.0,
            "reconstruction": 0.5,
            "consistency": 0.3,
            "vmf": 0.1
        }
        
        self.loss_weights = loss_weights or default_weights
        self.adaptive_weighting = adaptive_weighting
        
        # 损失函数
        self.contrastive_loss = MultiModalContrastiveLoss()
        self.reconstruction_loss = ReconstructionLoss()
        self.consistency_loss = ConsistencyLoss()
        self.vmf_loss = VonMisesFisherLoss(dim=64)
        
        # 自适应权重 (如果启用)
        if adaptive_weighting:
            self.adaptive_weights = nn.Parameter(
                torch.tensor(list(self.loss_weights.values()))
            )
    
    def forward(
        self,
        model_outputs: Dict[str, Any],
        targets: Dict[str, Any],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """计算总损失"""
        all_losses = {}
        
        # 1. 对比损失
        if "projected_embeddings" in model_outputs:
            contrastive_losses = self.contrastive_loss(
                image_embeddings=model_outputs["projected_embeddings"],
                text_embeddings=model_outputs.get("text_embeddings"),
                **kwargs
            )
            all_losses.update(contrastive_losses)
        
        # 2. 重建损失
        if "reconstructions" in model_outputs and "multimodal_data" in targets:
            reconstruction_losses = self.reconstruction_loss(
                reconstructed=model_outputs["reconstructions"],
                target=targets["multimodal_data"]
            )
            all_losses.update(reconstruction_losses)
        
        # 3. 一致性损失 (如果有teacher-student输出)
        if "teacher_outputs" in model_outputs and "student_outputs" in model_outputs:
            consistency_losses = self.consistency_loss(
                teacher_outputs=model_outputs["teacher_outputs"],
                student_outputs=model_outputs["student_outputs"]
            )
            all_losses.update(consistency_losses)
        
        # 4. vMF损失
        if "embeddings" in model_outputs and "target_embeddings" in targets:
            vmf_loss = self.vmf_loss(
                model_outputs["embeddings"],
                targets["target_embeddings"]
            )
            all_losses["vmf_loss"] = vmf_loss
        
        # 计算加权总损失
        total_loss = 0.0
        
        if self.adaptive_weighting:
            weights = F.softmax(self.adaptive_weights, dim=0)
        else:
            weights = torch.tensor(list(self.loss_weights.values()))
        
        loss_components = [
            all_losses.get("total_contrastive_loss", torch.tensor(0.0)),
            all_losses.get("total_reconstruction_loss", torch.tensor(0.0)),
            all_losses.get("total_consistency_loss", torch.tensor(0.0)),
            all_losses.get("vmf_loss", torch.tensor(0.0))
        ]
        
        for weight, loss_component in zip(weights, loss_components):
            total_loss += weight * loss_component
        
        all_losses["total_loss"] = total_loss
        
        return all_losses