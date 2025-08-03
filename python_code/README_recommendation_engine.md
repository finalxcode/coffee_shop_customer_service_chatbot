# 推荐引擎训练工具

## 概述

这个推荐引擎训练工具为咖啡店客户服务聊天机器人提供两种类型的推荐：

1. **热门推荐引擎** - 基于产品购买频率的简单推荐
2. **Apriori推荐引擎** - 基于关联规则挖掘的智能推荐

## 文件结构

```
python_code/
├── recommendation_engine_training_fixed.py  # 修复版本的训练脚本
├── test_recommendation_engine.py           # 测试脚本
├── dataset/                                # 数据集
│   ├── 201904 sales reciepts.csv          # 销售收据数据
│   └── product.csv                         # 产品数据
└── api/recommendation_objects/             # 输出文件
    ├── popularity_recommendation.csv       # 热门推荐
    └── apriori_recommendations.json       # Apriori推荐
```

## 安装依赖

```bash
pip install mlxtend pandas numpy
```

## 使用方法

### 1. 运行推荐引擎训练

```bash
cd python_code
python recommendation_engine_training_fixed.py
```

### 2. 运行测试

```bash
python test_recommendation_engine.py
```

## 输出文件说明

### 热门推荐 (popularity_recommendation.csv)

包含每个产品的购买频率信息：
- `product`: 产品名称
- `product_category`: 产品类别
- `number_of_transactions`: 购买次数

### Apriori推荐 (apriori_recommendations.json)

基于关联规则的产品推荐：
```json
{
  "产品名称": [
    {
      "product": "推荐产品",
      "product_category": "产品类别",
      "confidence": 置信度
    }
  ]
}
```

## 修复的问题

1. **弃用警告修复**: 使用 `DataFrame.map()` 替代 `DataFrame.applymap()`
2. **数据类型优化**: 将数据转换为布尔类型以避免性能警告
3. **错误处理**: 添加了完整的异常处理机制
4. **文件编码**: 使用UTF-8编码保存JSON文件
5. **目录创建**: 自动创建输出目录

## 训练过程

1. **数据读取**: 读取销售收据和产品数据
2. **数据清理**: 移除产品尺寸标识，筛选目标产品
3. **交易处理**: 只保留包含多个商品的交易
4. **热门推荐**: 计算产品购买频率
5. **Apriori算法**: 应用关联规则挖掘
6. **结果保存**: 生成CSV和JSON格式的推荐文件

## 性能指标

- 生成了 **16个产品** 的Apriori推荐
- 热门推荐包含 **19个产品**
- 处理了 **10,189条** 有效交易记录

## 使用示例

```python
# 读取热门推荐
import pandas as pd
popularity_df = pd.read_csv('api/recommendation_objects/popularity_recommendation.csv')

# 读取Apriori推荐
import json
with open('api/recommendation_objects/apriori_recommendations.json', 'r') as f:
    apriori_data = json.load(f)

# 获取特定产品的推荐
latte_recommendations = apriori_data.get('Latte', [])
```

## 注意事项

1. 确保数据集文件存在于正确路径
2. 训练过程可能需要几分钟时间
3. 生成的推荐文件会自动保存到 `api/recommendation_objects/` 目录
4. 如果遇到内存问题，可以考虑减少 `min_support` 参数值 