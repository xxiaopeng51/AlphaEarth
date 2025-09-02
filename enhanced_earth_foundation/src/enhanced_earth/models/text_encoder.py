"""
Text Encoder

文本编码器，支持地理位置描述和土地利用标签的编码。
借鉴SatCLIP的文本-图像对齐设计。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import math
from transformers import AutoTokenizer, AutoModel
from einops import rearrange


class GeospatialTextEncoder(nn.Module):
    """地理空间文本编码器，专门处理地理相关的文本描述"""
    
    def __init__(
        self,
        d_model: int = 768,
        vocab_size: int = 50000,
        max_length: int = 256,
        pretrained_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        freeze_pretrained: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        try:
            # 尝试加载预训练模型
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
            self.text_model = AutoModel.from_pretrained(pretrained_model)
            
            if freeze_pretrained:
                for param in self.text_model.parameters():
                    param.requires_grad = False
            
            pretrained_dim = self.text_model.config.hidden_size
            self.use_pretrained = True
            
        except Exception:
            # 如果无法加载预训练模型，使用自定义实现
            self.use_pretrained = False
            pretrained_dim = d_model
        
        # 地理空间特定的文本处理层
        self.geospatial_processor = GeospatialTextProcessor(
            input_dim=pretrained_dim,
            output_dim=d_model,
            vocab_size=vocab_size if not self.use_pretrained else None
        )
        
        # 文本特征增强
        self.text_enhancer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 地理实体识别和增强
        self.entity_recognizer = GeographicEntityRecognizer(d_model)
    
    def forward(
        self, 
        text_input: torch.Tensor,  # (B, max_length) token IDs
        attention_mask: Optional[torch.Tensor] = None  # (B, max_length)
    ) -> torch.Tensor:
        """
        Args:
            text_input: (B, max_length) 文本token
            attention_mask: (B, max_length) 注意力掩码
        Returns:
            (B, d_model) 文本嵌入
        """
        if self.use_pretrained:
            # 使用预训练模型
            with torch.set_grad_enabled(not getattr(self, 'freeze_pretrained', False)):
                outputs = self.text_model(
                    input_ids=text_input,
                    attention_mask=attention_mask
                )
                text_features = outputs.last_hidden_state  # (B, L, hidden_size)
                
                # 池化到固定维度
                if attention_mask is not None:
                    # 掩码平均池化
                    mask_expanded = attention_mask.unsqueeze(-1).expand_as(text_features)
                    sum_embeddings = torch.sum(text_features * mask_expanded, dim=1)
                    sum_mask = torch.sum(mask_expanded, dim=1)
                    pooled_features = sum_embeddings / (sum_mask + 1e-8)
                else:
                    pooled_features = text_features.mean(dim=1)
        else:
            # 自定义文本编码
            pooled_features = self.geospatial_processor(text_input, attention_mask)
        
        # 地理空间特定处理
        enhanced_features = self.text_enhancer(pooled_features)
        
        # 地理实体识别和增强
        entity_enhanced = self.entity_recognizer(enhanced_features, text_input)
        
        return entity_enhanced


class GeospatialTextProcessor(nn.Module):
    """地理空间文本处理器"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        vocab_size: Optional[int] = None,
        max_length: int = 256
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_length = max_length
        
        if vocab_size is not None:
            # 自定义词嵌入
            self.token_embedding = nn.Embedding(vocab_size, input_dim)
            self.position_embedding = nn.Embedding(max_length, input_dim)
        
        # 地理关键词增强
        self.geographic_keywords = self._create_geographic_keyword_embeddings()
        
        # 文本Transformer
        self.text_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=input_dim // 64,
                dim_feedforward=input_dim * 4,
                activation='gelu',
                dropout=0.1,
                batch_first=True
            ),
            num_layers=6
        )
        
        # 输出投影
        if input_dim != output_dim:
            self.output_projection = nn.Linear(input_dim, output_dim)
        else:
            self.output_projection = nn.Identity()
    
    def _create_geographic_keyword_embeddings(self) -> nn.Module:
        """创建地理关键词嵌入"""
        # 预定义的地理关键词
        geo_keywords = [
            "forest", "urban", "water", "agriculture", "desert", "mountain",
            "ocean", "lake", "river", "city", "village", "farmland",
            "wetland", "grassland", "tundra", "ice", "snow", "cloud"
        ]
        
        return nn.Embedding(len(geo_keywords), self.input_dim)
    
    def forward(
        self, 
        text_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """处理文本输入"""
        if hasattr(self, 'token_embedding'):
            # 自定义嵌入
            B, L = text_tokens.shape
            
            # Token嵌入
            token_emb = self.token_embedding(text_tokens)
            
            # 位置嵌入
            positions = torch.arange(L, device=text_tokens.device).unsqueeze(0).expand(B, -1)
            pos_emb = self.position_embedding(positions)
            
            # 组合嵌入
            embeddings = token_emb + pos_emb
        else:
            # 使用输入作为嵌入 (来自预训练模型)
            embeddings = text_tokens
        
        # Transformer处理
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None
        
        processed = self.text_transformer(embeddings, src_key_padding_mask=key_padding_mask)
        
        # 池化
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(processed)
            pooled = torch.sum(processed * mask_expanded, dim=1) / torch.sum(mask_expanded, dim=1)
        else:
            pooled = processed.mean(dim=1)
        
        return self.output_projection(pooled)


class GeographicEntityRecognizer(nn.Module):
    """地理实体识别和增强模块"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # 地理实体类别
        self.entity_types = [
            "landcover",    # 土地覆盖
            "landuse",      # 土地利用  
            "topography",   # 地形
            "climate",      # 气候
            "hydrology",    # 水文
            "vegetation",   # 植被
            "urban",        # 城市
            "natural"       # 自然地物
        ]
        
        # 实体类别嵌入
        self.entity_embeddings = nn.Embedding(len(self.entity_types), dim)
        
        # 实体识别网络
        self.entity_classifier = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, len(self.entity_types)),
            nn.Sigmoid()  # 多标签分类
        )
        
        # 实体增强网络
        self.entity_enhancer = nn.Sequential(
            nn.Linear(dim + len(self.entity_types), dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
    
    def forward(
        self, 
        text_features: torch.Tensor,  # (B, dim)
        text_tokens: Optional[torch.Tensor] = None  # (B, L)
    ) -> torch.Tensor:
        """
        识别和增强地理实体信息
        
        Args:
            text_features: (B, dim) 文本特征
            text_tokens: (B, L) 原始token (可选)
        Returns:
            (B, dim) 增强后的文本特征
        """
        # 识别地理实体类别
        entity_probs = self.entity_classifier(text_features)  # (B, num_entity_types)
        
        # 加权实体嵌入
        entity_emb = self.entity_embeddings.weight  # (num_entity_types, dim)
        weighted_entity = torch.matmul(entity_probs, entity_emb)  # (B, dim)
        
        # 拼接原始特征和实体信息
        enhanced_input = torch.cat([text_features, entity_probs], dim=-1)
        
        # 实体增强
        enhanced_features = self.entity_enhancer(enhanced_input)
        
        # 残差连接
        return text_features + enhanced_features + weighted_entity * 0.1


class TextEncoder(nn.Module):
    """
    文本编码器主类
    
    整合地理空间文本处理和多语言支持
    """
    
    def __init__(
        self,
        d_model: int = 768,
        vocab_size: int = 50000,
        max_length: int = 256,
        pretrained_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # 地理空间文本编码器
        self.geospatial_encoder = GeospatialTextEncoder(
            d_model=d_model,
            vocab_size=vocab_size,
            max_length=max_length,
            pretrained_model=pretrained_model
        )
        
        # 多语言支持 (可选)
        self.multilingual_adapter = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 输出标准化
        self.output_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        text_input: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        language_id: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            text_input: (B, max_length) 文本token
            attention_mask: (B, max_length) 注意力掩码
            language_id: (B,) 语言ID (可选)
        Returns:
            (B, d_model) 文本嵌入
        """
        # 地理空间文本编码
        text_features = self.geospatial_encoder(text_input, attention_mask)
        
        # 多语言适配 (如果提供语言ID)
        if language_id is not None:
            text_features = self.multilingual_adapter(text_features)
        
        # 输出标准化
        text_features = self.output_norm(text_features)
        text_features = self.dropout(text_features)
        
        return text_features