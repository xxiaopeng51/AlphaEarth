"""可重现性工具"""

import torch
import numpy as np
import random
import os


def set_deterministic(seed: int = 42):
    """设置确定性训练环境"""
    # Python随机种子
    random.seed(seed)
    
    # NumPy随机种子
    np.random.seed(seed)
    
    # PyTorch随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Deterministic mode enabled with seed: {seed}")