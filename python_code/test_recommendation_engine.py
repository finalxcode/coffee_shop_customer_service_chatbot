#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试推荐引擎脚本
验证修复后的推荐引擎是否正常工作
"""

import os
import json
import pandas as pd
from recommendation_engine_training_fixed import main

def test_recommendation_engine():
    """测试推荐引擎功能"""
    print("🧪 开始测试推荐引擎...")
    
    # 检查必要文件是否存在
    required_files = [
        'dataset/201904 sales reciepts.csv',
        'dataset/product.csv'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ 缺少必要文件: {file_path}")
            return False
        else:
            print(f"✅ 文件存在: {file_path}")
    
    # 运行推荐引擎训练
    print("\n🚀 开始训练推荐引擎...")
    success = main()
    
    if not success:
        print("❌ 推荐引擎训练失败")
        return False
    
    # 验证输出文件
    output_files = [
        'api/recommendation_objects/popularity_recommendation.csv',
        'api/recommendation_objects/apriori_recommendations.json',
        'rules_basket.pkl'
    ]
    
    print("\n📁 验证输出文件...")
    for file_path in output_files:
        if os.path.exists(file_path):
            print(f"✅ 输出文件存在: {file_path}")
            
            # 检查文件内容
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                print(f"   - 包含 {len(df)} 行数据")
            elif file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"   - 包含 {len(data)} 个产品推荐")
        else:
            print(f"❌ 输出文件缺失: {file_path}")
            return False
    
    print("\n🎉 所有测试通过！推荐引擎工作正常。")
    return True

def show_sample_recommendations():
    """显示示例推荐"""
    print("\n📊 示例推荐:")
    
    # 显示热门推荐
    try:
        popularity_df = pd.read_csv('api/recommendation_objects/popularity_recommendation.csv')
        print("\n🔥 热门推荐 (前5名):")
        print(popularity_df.head().to_string(index=False))
    except Exception as e:
        print(f"无法读取热门推荐: {e}")
    
    # 显示Apriori推荐
    try:
        with open('api/recommendation_objects/apriori_recommendations.json', 'r', encoding='utf-8') as f:
            apriori_data = json.load(f)
        
        print(f"\n🔄 Apriori推荐 (共{len(apriori_data)}个产品):")
        for i, (product, recommendations) in enumerate(list(apriori_data.items())[:3]):
            print(f"\n产品: {product}")
            for rec in recommendations[:3]:  # 只显示前3个推荐
                print(f"  - {rec['product']} ({rec['product_category']}) - 置信度: {rec['confidence']:.3f}")
    except Exception as e:
        print(f"无法读取Apriori推荐: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("推荐引擎测试工具")
    print("=" * 50)
    
    # 运行测试
    if test_recommendation_engine():
        # 显示示例推荐
        show_sample_recommendations()
        print("\n✅ 推荐引擎测试完成！")
    else:
        print("\n❌ 推荐引擎测试失败！") 