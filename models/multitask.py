"""
Multi-task frame monitoring models with different backbone architectures.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
from transformers import (
    LongformerModel, LongformerConfig,
    BigBirdModel, BigBirdConfig,
    AutoModel, AutoConfig
)


class FrameMonitoringModel(nn.Module):
    """
    Multi-task model for frame intensity regression and presence classification.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        num_frames: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        
        # Regression head for frame intensity
        self.regression_head = nn.Linear(hidden_size, num_frames)
        
        # Classification head for frame presence
        self.classification_head = nn.Linear(hidden_size, num_frames)
        
        self.num_frames = num_frames
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        global_attention_mask: Optional[torch.LongTensor] = None,
        y_reg: Optional[torch.FloatTensor] = None,
        y_cls: Optional[torch.FloatTensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [B, L]
            attention_mask: Attention mask [B, L]
            global_attention_mask: Global attention mask for Longformer [B, L]
            y_reg: Regression targets [B, F]
            y_cls: Classification targets [B, F]
            
        Returns:
            Dictionary with predictions and losses
        """
        # Forward through backbone
        if global_attention_mask is not None:
            # Longformer
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask
            )
        else:
            # Other models
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Get CLS representation
        cls_output = outputs.last_hidden_state[:, 0]  # [B, H]
        cls_output = self.dropout(cls_output)
        
        # Predictions
        reg_pred = self.regression_head(cls_output)  # [B, F]
        cls_logit = self.classification_head(cls_output)  # [B, F]
        cls_prob = torch.sigmoid(cls_logit)  # [B, F]
        
        result = {
            'reg_pred': reg_pred,
            'cls_logit': cls_logit,
            'cls_prob': cls_prob
        }
        
        # Compute losses if targets provided
        if y_reg is not None and y_cls is not None:
            from ..training.losses import compute_multitask_loss
            
            # Default loss configuration
            total_loss, loss_reg, loss_cls = compute_multitask_loss(
                reg_pred=reg_pred,
                y_reg=y_reg,
                cls_logit=cls_logit,
                y_cls=y_cls,
                reg_loss="huber",
                cls_loss="bce",
                reg_weight=1.0,
                cls_weight=1.0
            )
            
            result.update({
                'loss': total_loss,
                'loss_reg': loss_reg,
                'loss_cls': loss_cls
            })
        
        return result


def build_longformer_model(
    pretrained_name: str,
    num_frames: int,
    dropout: float = 0.1,
    global_attention: str = "cls",
    num_global_tokens: int = 1
) -> FrameMonitoringModel:
    """
    Build Longformer-based frame monitoring model.
    
    Args:
        pretrained_name: Pretrained model name
        num_frames: Number of frame types
        dropout: Dropout rate
        global_attention: Global attention strategy
        num_global_tokens: Number of global attention tokens
        
    Returns:
        FrameMonitoringModel instance
    """
    config = LongformerConfig.from_pretrained(pretrained_name)
    backbone = LongformerModel.from_pretrained(pretrained_name, config=config)
    
    model = FrameMonitoringModel(
        backbone=backbone,
        hidden_size=config.hidden_size,
        num_frames=num_frames,
        dropout=dropout
    )
    
    return model


def build_bigbird_model(
    pretrained_name: str,
    num_frames: int,
    dropout: float = 0.1
) -> FrameMonitoringModel:
    """
    Build BigBird-based frame monitoring model.
    
    Args:
        pretrained_name: Pretrained model name
        num_frames: Number of frame types
        dropout: Dropout rate
        
    Returns:
        FrameMonitoringModel instance
    """
    config = BigBirdConfig.from_pretrained(pretrained_name)
    backbone = BigBirdModel.from_pretrained(pretrained_name, config=config)
    
    model = FrameMonitoringModel(
        backbone=backbone,
        hidden_size=config.hidden_size,
        num_frames=num_frames,
        dropout=dropout
    )
    
    return model


def build_hat_model(
    pretrained_name: str,
    num_frames: int,
    dropout: float = 0.1,
    max_sentences: int = 64,
    max_sentence_length: int = 128
) -> FrameMonitoringModel:
    """
    Build Hierarchical Attention Transformer (HAT) model.
    
    Args:
        pretrained_name: Pretrained model name for sentence encoder
        num_frames: Number of frame types
        dropout: Dropout rate
        max_sentences: Maximum number of sentences
        max_sentence_length: Maximum sentence length
        
    Returns:
        FrameMonitoringModel instance
    """
    # This is a simplified HAT implementation
    # In practice, you'd want a more sophisticated hierarchical architecture
    
    config = AutoConfig.from_pretrained(pretrained_name)
    sentence_encoder = AutoModel.from_pretrained(pretrained_name, config=config)
    
    class HATBackbone(nn.Module):
        def __init__(self, sentence_encoder, hidden_size):
            super().__init__()
            self.sentence_encoder = sentence_encoder
            self.sentence_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.layer_norm = nn.LayerNorm(hidden_size)
            
        def forward(self, input_ids, attention_mask, **kwargs):
            # This is a simplified implementation
            # In practice, you'd split into sentences and process hierarchically
            outputs = self.sentence_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Simple pooling for now
            sequence_output = outputs.last_hidden_state
            pooled_output = sequence_output.mean(dim=1)
            
            # Fake the expected output format
            class SimpleOutput:
                def __init__(self, last_hidden_state):
                    self.last_hidden_state = last_hidden_state
            
            # Add batch dimension back for CLS token extraction
            fake_cls = pooled_output.unsqueeze(1)  # [B, 1, H]
            return SimpleOutput(fake_cls)
    
    backbone = HATBackbone(sentence_encoder, config.hidden_size)
    
    model = FrameMonitoringModel(
        backbone=backbone,
        hidden_size=config.hidden_size,
        num_frames=num_frames,
        dropout=dropout
    )
    
    return model


def build_model(
    backbone: str,
    pretrained_name: str,
    num_frames: int,
    dropout: float = 0.1,
    **kwargs
) -> FrameMonitoringModel:
    """
    Build model based on backbone type.
    
    Args:
        backbone: Model backbone type
        pretrained_name: Pretrained model name
        num_frames: Number of frame types
        dropout: Dropout rate
        **kwargs: Additional arguments
        
    Returns:
        FrameMonitoringModel instance
    """
    if backbone == "longformer":
        return build_longformer_model(
            pretrained_name=pretrained_name,
            num_frames=num_frames,
            dropout=dropout,
            global_attention=kwargs.get("global_attention", "cls"),
            num_global_tokens=kwargs.get("num_global_tokens", 1)
        )
    elif backbone == "bigbird":
        return build_bigbird_model(
            pretrained_name=pretrained_name,
            num_frames=num_frames,
            dropout=dropout
        )
    elif backbone == "hat":
        return build_hat_model(
            pretrained_name=pretrained_name,
            num_frames=num_frames,
            dropout=dropout,
            max_sentences=kwargs.get("max_sentences", 64),
            max_sentence_length=kwargs.get("max_sentence_length", 128)
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")