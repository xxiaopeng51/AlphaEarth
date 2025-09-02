"""评估指标"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, f1_score


class EvaluationMetrics:
    """评估指标计算器"""
    
    def __init__(self):
        self.metrics = [
            "mse", "mae", "ssim", "psnr", "cosine_similarity"
        ]
    
    def compute_metrics(
        self, 
        outputs: Dict[str, torch.Tensor], 
        targets: Dict[str, Any]
    ) -> Dict[str, float]:
        """计算所有指标"""
        metrics = {}
        
        # 嵌入质量指标
        if "embeddings" in outputs:
            emb_metrics = self._compute_embedding_metrics(outputs["embeddings"])
            metrics.update(emb_metrics)
        
        # 重建质量指标
        if "reconstructions" in outputs and "multimodal_data" in targets:
            recon_metrics = self._compute_reconstruction_metrics(
                outputs["reconstructions"], 
                targets["multimodal_data"]
            )
            metrics.update(recon_metrics)
        
        return metrics
    
    def _compute_embedding_metrics(self, embeddings: torch.Tensor) -> Dict[str, float]:
        """计算嵌入质量指标"""
        metrics = {}
        
        # 检查单位向量性质
        norms = torch.norm(embeddings, p=2, dim=-1)
        metrics["embedding_norm_mean"] = norms.mean().item()
        metrics["embedding_norm_std"] = norms.std().item()
        
        # 嵌入多样性
        flat_emb = embeddings.view(-1, embeddings.shape[-1])
        pairwise_sim = torch.mm(flat_emb, flat_emb.t())
        metrics["embedding_diversity"] = (1 - pairwise_sim.mean()).item()
        
        return metrics
    
    def _compute_reconstruction_metrics(
        self, 
        reconstructions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """计算重建质量指标"""
        metrics = {}
        
        for modality in reconstructions.keys():
            if modality in targets:
                recon = reconstructions[modality]
                target = targets[modality]
                
                # MSE
                mse = F.mse_loss(recon, target).item()
                metrics[f"{modality}_mse"] = mse
                
                # MAE
                mae = F.l1_loss(recon, target).item()
                metrics[f"{modality}_mae"] = mae
                
                # PSNR
                psnr = self._compute_psnr(recon, target)
                metrics[f"{modality}_psnr"] = psnr
        
        return metrics
    
    def _compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """计算PSNR"""
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return 100.0
        
        max_val = target.max()
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
        return psnr.item()