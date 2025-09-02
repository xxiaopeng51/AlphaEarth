#!/usr/bin/env python3
"""
Enhanced Earth Foundation Model Training Script

主训练脚本，支持单机和分布式训练。

Usage:
    # 单机训练
    python scripts/train.py --config configs/global_model.yaml
    
    # 分布式训练
    torchrun --nproc_per_node=4 scripts/train.py --config configs/global_model.yaml --distributed
    
    # 恢复训练
    python scripts/train.py --config configs/global_model.yaml --resume checkpoints/epoch_10.pt
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from enhanced_earth.models.enhanced_earth_model import EnhancedEarthFoundationModel
from enhanced_earth.training.trainer import EnhancedEarthTrainer, TrainingConfig
from enhanced_earth.utils.logging import setup_logging
from enhanced_earth.utils.reproducibility import set_deterministic


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Train Enhanced Earth Foundation Model")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training"
    )
    
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank for distributed training"
    )
    
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="World size for distributed training"
    )
    
    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="Master address for distributed training"
    )
    
    parser.add_argument(
        "--master_port",
        type=str,
        default="12355",
        help="Master port for distributed training"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    return parser.parse_args()


def setup_distributed(args):
    """设置分布式训练环境"""
    if args.distributed:
        # 设置环境变量
        os.environ["MASTER_ADDR"] = args.master_addr
        os.environ["MASTER_PORT"] = args.master_port
        os.environ["WORLD_SIZE"] = str(args.world_size)
        os.environ["RANK"] = str(args.local_rank)
        
        # 初始化进程组
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=args.world_size,
            rank=args.local_rank
        )
        
        torch.cuda.set_device(args.local_rank)
        
        print(f"Distributed training initialized: rank {args.local_rank}/{args.world_size}")


def create_model_from_config(config: DictConfig) -> EnhancedEarthFoundationModel:
    """从配置创建模型"""
    model_config = config.model
    input_modalities = config.input_modalities
    
    # 转换模态配置格式
    modalities_dict = {}
    for name, mod_config in input_modalities.items():
        modalities_dict[name] = {
            "channels": mod_config.channels,
            "resolution": mod_config.resolution
        }
    
    model = EnhancedEarthFoundationModel(
        model_size=model_config.model_size,
        input_modalities=modalities_dict,
        embed_dim=model_config.embed_dim,
        enable_text_alignment=model_config.enable_text_alignment,
        enable_reconstruction=model_config.enable_reconstruction,
        dropout=model_config.dropout
    )
    
    return model


def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志
    setup_logging(debug=args.debug)
    
    # 设置分布式训练
    setup_distributed(args)
    
    # 加载配置
    config = OmegaConf.load(args.config)
    
    # 更新分布式设置
    if args.distributed:
        config.training.distributed = True
        config.training.world_size = args.world_size
        config.training.rank = args.local_rank
    
    # 设置确定性训练
    if config.get("deterministic", False):
        set_deterministic(config.training.get("seed", 42))
    
    # 创建模型
    model = create_model_from_config(config)
    
    # 创建训练配置
    training_config = TrainingConfig(
        **config.training,
        resume_from=args.resume
    )
    
    # 创建训练器
    trainer = EnhancedEarthTrainer(
        config=training_config,
        model=model,
        data_config=config.input_modalities
    )
    
    try:
        # 开始训练
        final_metrics = trainer.train()
        
        # 输出最终结果
        if not args.distributed or args.local_rank == 0:
            print("Training completed successfully!")
            print("Final metrics:")
            for key, value in final_metrics.items():
                print(f"  {key}: {value:.4f}")
            
            # 保存最终模型
            trainer.export_model(
                f"{config.training.save_dir}/final_model.pt",
                format="pytorch"
            )
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
        if not args.distributed or args.local_rank == 0:
            trainer.save_checkpoint("interrupted")
    
    except Exception as e:
        print(f"Training failed with error: {e}")
        if not args.distributed or args.local_rank == 0:
            trainer.save_checkpoint("error")
        raise
    
    finally:
        # 清理分布式训练
        if args.distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()