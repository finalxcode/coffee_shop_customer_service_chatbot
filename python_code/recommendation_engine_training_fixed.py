#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版本的推荐引擎训练脚本
解决弃用警告和其他潜在问题
"""

import pandas as pd
import numpy as np
import warnings
import json
import os
from mlxtend.frequent_patterns import association_rules, apriori

# 忽略弃用警告
warnings.filterwarnings('ignore', category=DeprecationWarning)

def main():
    """主函数：执行推荐引擎训练"""
    try:
        print("开始推荐引擎训练...")
        
        # 1. 读取数据集
        print("正在读取数据集...")
        sales_reciepts = pd.read_csv('dataset/201904 sales reciepts.csv')
        product = pd.read_csv('dataset/product.csv')
        
        print(f"销售收据数据形状: {sales_reciepts.shape}")
        print(f"产品数据形状: {product.shape}")
        
        # 2. 数据合并
        print("正在合并数据...")
        dataset = pd.merge(
            sales_reciepts[['transaction_id', 'transaction_date', 'sales_outlet_id', 'customer_id', 'product_id', 'quantity']],
            product[['product_id', 'product_category', 'product']],
            on='product_id', 
            how='left'
        )
        
        # 3. 清理产品名称（移除尺寸标识）
        print("正在清理产品名称...")
        dataset['product'] = dataset['product'].str.replace(' Rg', '')
        dataset['product'] = dataset['product'].str.replace(' Sm', '')
        dataset['product'] = dataset['product'].str.replace(' Lg', '')
        
        # 4. 选择产品子集
        products_to_take = [
            'Cappuccino', 'Latte', 'Espresso shot',
            'Dark chocolate', 'Sugar Free Vanilla syrup', 'Chocolate syrup',
            'Carmel syrup', 'Hazelnut syrup', 'Ginger Scone',
            'Chocolate Croissant', 'Jumbo Savory Scone', 'Cranberry Scone', 'Hazelnut Biscotti',
            'Croissant', 'Almond Croissant', 'Oatmeal Scone', 'Chocolate Chip Biscotti',
            'Ginger Biscotti',
        ]
        
        dataset = dataset[dataset['product'].isin(products_to_take)]
        print(f"筛选后的数据集形状: {dataset.shape}")
        
        # 5. 清理交易数据
        print("正在清理交易数据...")
        dataset['transaction'] = dataset['transaction_id'].astype(str) + "_" + dataset['customer_id'].astype(str)
        
        # 只保留包含多个商品的交易
        num_of_items_for_each_transaction = dataset['transaction'].value_counts().reset_index()
        valid_transactions = num_of_items_for_each_transaction[num_of_items_for_each_transaction['count'] > 1]['transaction'].tolist()
        dataset = dataset[dataset['transaction'].isin(valid_transactions)]
        
        print(f"有效交易后的数据集形状: {dataset.shape}")
        
        # 6. 创建热门推荐引擎
        print("正在创建热门推荐引擎...")
        popularity_recommendation = dataset.groupby(['product', 'product_category']).count().reset_index()
        popularity_recommendation = popularity_recommendation[['product', 'product_category', 'transaction_id']]
        popularity_recommendation = popularity_recommendation.rename(columns={'transaction_id': 'number_of_transactions'})
        
        # 确保输出目录存在
        os.makedirs('api/recommendation_objects', exist_ok=True)
        popularity_recommendation.to_csv('api/recommendation_objects/popularity_recommendation.csv', index=False)
        print("热门推荐已保存")
        
        # 7. 创建Apriori推荐引擎
        print("正在创建Apriori推荐引擎...")
        train_basket = (dataset.groupby(['transaction', 'product'])['product'].count().reset_index(name='Count'))
        
        # 创建透视表
        my_basket = train_basket.pivot_table(
            index='transaction', 
            columns='product', 
            values='Count', 
            aggfunc='sum'
        ).fillna(0)
        
        # 编码函数（修复弃用警告）
        def encode_units(x):
            return 1 if x > 0 else 0
        
        # 使用map替代applymap以避免弃用警告
        my_basket_sets = my_basket.map(encode_units)
        
        # 转换为布尔类型以避免弃用警告
        my_basket_sets = my_basket_sets.astype(bool)
        
        # 应用Apriori算法
        frequent_items = apriori(my_basket_sets, min_support=0.05, use_colnames=True)
        
        # 生成关联规则
        rules_basket = association_rules(frequent_items, metric="lift", min_threshold=1)
        
        # 保存规则
        rules_basket.to_pickle('rules_basket.pkl')
        print("Apriori规则已保存")
        
        # 8. 创建JSON格式的推荐
        print("正在创建JSON格式推荐...")
        product_categories = dataset[['product', 'product_category']].drop_duplicates().set_index('product').to_dict()['product_category']
        
        recommendations_json = {}
        
        antecedents = rules_basket['antecedents'].unique()
        for antecedent in antecedents:
            df_rec = rules_basket[rules_basket['antecedents'] == antecedent]
            df_rec = df_rec.sort_values('confidence', ascending=False)
            key = "_".join(antecedent)
            recommendations_json[key] = []
            
            for _, row in df_rec.iterrows():
                rec_objects = row['consequents']
                for rec_object in rec_objects:
                    # 检查是否已存在
                    already_exists = False
                    for current_rec_object in recommendations_json[key]:
                        if rec_object == current_rec_object['product']:
                            already_exists = True
                            break
                    
                    if already_exists:
                        continue
                    
                    rec = {
                        'product': rec_object,
                        'product_category': product_categories.get(rec_object, 'Unknown'),
                        'confidence': float(row['confidence'])
                    }
                    recommendations_json[key].append(rec)
        
        # 保存JSON推荐
        with open('api/recommendation_objects/apriori_recommendations.json', 'w', encoding='utf-8') as json_file:
            json.dump(recommendations_json, json_file, ensure_ascii=False, indent=2)
        
        print("推荐引擎训练完成！")
        print(f"生成了 {len(recommendations_json)} 个产品的推荐")
        print(f"热门推荐包含 {len(popularity_recommendation)} 个产品")
        
        return True
        
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ 推荐引擎训练成功完成！")
    else:
        print("❌ 推荐引擎训练失败！") 