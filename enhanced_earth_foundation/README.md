# Enhanced Earth Foundation Model

基于Google's AlphaEarth Foundations的改进版全球尺度多模态地球基础模型，融合了Clay Foundation Model、SatCLIP、Prithvi等先进模型的最佳实践。

## 核心特性

### 1. 多模态数据支持
- **光学影像**: Sentinel-2, Landsat 8/9, MODIS
- **雷达数据**: Sentinel-1 SAR, PALSAR-2
- **高光谱**: PRISMA, EnMAP
- **激光雷达**: GEDI, ICESat-2
- **环境数据**: ERA5-Land, DEM, 土地覆盖
- **文本描述**: 地理位置描述、土地利用标签

### 2. 先进架构设计
- **多尺度STP编码器**: 改进的Space-Time-Precision架构
- **动态多模态融合**: 基于注意力的自适应融合机制
- **位置感知编码**: 结合地理坐标和时间的高精度编码
- **缩放法则优化**: 支持从小型到超大型模型的平滑扩展

### 3. 全球覆盖能力
- **连续时间支持**: 支持任意时间间隔的插值和外推
- **跨区域泛化**: 在全球不同地理区域保持稳定性能
- **多分辨率处理**: 从10m到1km的多尺度数据处理

## 模型架构

```
输入多模态数据 → 源编码器 → 多尺度STP编码器 → 时间汇聚 → 64D球面嵌入
                    ↓
              对比学习 ← 文本编码器 ← 地理文本描述
                    ↓  
              解码器 → 重建各模态数据
```

## 安装和使用

```bash
# 克隆仓库
git clone <repository_url>
cd enhanced_earth_foundation

# 安装依赖
pip install -r requirements.txt

# 安装包
pip install -e .

# 运行训练
python scripts/train.py --config configs/global_model.yaml
```

## 技术亮点

1. **改进的STP架构**: 相比原始AlphaEarth，增加了跨模态信息交换
2. **动态嵌入**: 借鉴Clay的动态嵌入，支持任意波段配置
3. **位置编码**: 结合SatCLIP的地理位置编码方法
4. **多模态对比学习**: 扩展的CLIP框架支持多种模态组合
5. **可扩展训练**: 支持分布式训练和模型并行

## 引用

如果您使用了这个模型，请引用相关论文：

```bibtex
@article{brown2025alphaearth,
  title={AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data},
  author={Brown, Christopher F and Kazmierski, Michal R and Pasquarella, Valerie J and others},
  journal={arXiv preprint arXiv:2507.22291},
  year={2025}
}
```