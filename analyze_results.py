# 更新analyze_results.py以显示更多能量相关信息

#!/usr/bin/env python
# analyze_results.py - 用于分析和导出量子回路计算结果

import os
import pickle
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit

def load_batch_results(directory):
    """加载目录中的所有批处理结果文件"""
    results = []
    batch_files = [f for f in os.listdir(directory) if f.startswith('batch_') and f.endswith('_results.pkl')]
    
    print(f"找到 {len(batch_files)} 个批处理结果文件")
    
    for batch_file in sorted(batch_files, key=lambda x: int(x.split('_')[1])):
        file_path = os.path.join(directory, batch_file)
        try:
            with open(file_path, 'rb') as f:
                batch_data = pickle.load(f)
                results.extend(batch_data["results"])
            print(f"已加载 {file_path}")
        except Exception as e:
            print(f"加载 {file_path} 失败: {e}")
    
    return results

def load_circuits(file_path):
    """加载量子回路文件"""
    try:
        with open(file_path, 'rb') as f:
            circuits = pickle.load(f)
        print(f"已加载 {len(circuits)} 个量子回路，来自 {file_path}")
        return circuits
    except Exception as e:
        print(f"加载 {file_path} 失败: {e}")
        return []

def results_to_dataframe(results):
    """将结果转换为DataFrame"""
    data = []
    
    for res in results:
        row = {
            'global_index': res['global_index'],
            'batch_index': res['batch_index'],
            'batch_inner_index': res['batch_inner_index'],
            'energy': res['optimal_value'],
            'expressibility': res['expressibility'],
            'status': res['status'],
            'time_taken': res['time_taken'],
            'num_qubits': res['circuit'].num_qubits if isinstance(res['circuit'], QuantumCircuit) else None,
            'depth': res['circuit'].depth() if isinstance(res['circuit'], QuantumCircuit) else None,
        }
        
        # 添加所有能量值（如果存在）
        if 'all_energies' in res and res['all_energies']:
            row['all_energies'] = res['all_energies']
            if len(res['all_energies']) > 1:
                row['energy_min'] = min(res['all_energies'])
                row['energy_max'] = max(res['all_energies'])
                row['energy_mean'] = sum(res['all_energies']) / len(res['all_energies'])
                row['energy_std'] = np.std(res['all_energies']) if len(res['all_energies']) > 1 else 0
        
        # 添加优化器评估次数（如果存在）
        if 'cost_function_evals' in res:
            row['cost_function_evals'] = res['cost_function_evals']
        
        data.append(row)
    
    return pd.DataFrame(data)

def visualize_data(df, output_dir):
    """生成数据可视化"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 能量分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(df['energy'].dropna(), bins=30, alpha=0.7)
    plt.xlabel('能量')
    plt.ylabel('频率')
    plt.title('能量分布')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'energy_distribution.png'), dpi=300)
    plt.close()
    
    # 如果有重复VQE运行的数据，绘制能量方差图
    if 'energy_std' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df['energy_std'].dropna(), bins=30, alpha=0.7)
        plt.xlabel('能量标准差')
        plt.ylabel('频率')
        plt.title('VQE重复运行的能量标准差分布')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'energy_std_distribution.png'), dpi=300)
        plt.close()
    
    # 如果有函数评估次数数据，绘制直方图
    if 'cost_function_evals' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df['cost_function_evals'].dropna(), bins=30, alpha=0.7)
        plt.xlabel('函数评估次数')
        plt.ylabel('频率')
        plt.title('VQE函数评估次数分布')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'cost_function_evals.png'), dpi=300)
        plt.close()
    
    # 如果有表达性数据，生成表达性分布图
    if not df['expressibility'].isna().all():
        plt.figure(figsize=(10, 6))
        plt.hist(df['expressibility'].dropna(), bins=30, alpha=0.7)
        plt.xlabel('表达性')
        plt.ylabel('频率')
        plt.title('表达性分布')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'expressibility_distribution.png'), dpi=300)
        plt.close()
        
        # 能量与表达性散点图
        plt.figure(figsize=(10, 6))
        valid_data = df.dropna(subset=['energy', 'expressibility'])
        plt.scatter(valid_data['energy'], valid_data['expressibility'], alpha=0.5)
        plt.xlabel('能量')
        plt.ylabel('表达性')
        plt.title('能量与表达性关系')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'energy_vs_expressibility.png'), dpi=300)
        plt.close()
    
    # 计算时间分布
    plt.figure(figsize=(10, 6))
    plt.hist(df['time_taken'].dropna(), bins=30, alpha=0.7)
    plt.xlabel('计算时间 (秒)')
    plt.ylabel('频率')
    plt.title('计算时间分布')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'calculation_time.png'), dpi=300)
    plt.close()
    
    # 回路深度与能量关系
    plt.figure(figsize=(10, 6))
    valid_data = df.dropna(subset=['depth', 'energy'])
    plt.scatter(valid_data['depth'], valid_data['energy'], alpha=0.5)
    plt.xlabel('回路深度')
    plt.ylabel('能量')
    plt.title('回路深度与能量关系')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'depth_vs_energy.png'), dpi=300)
    plt.close()
    
    # 生成统计摘要
    with open(os.path.join(output_dir, 'statistics.txt'), 'w') as f:
        f.write(f"数据统计摘要\n")
        f.write(f"=============\n\n")
        f.write(f"总回路数量: {len(df)}\n")
        f.write(f"成功计算数量: {df[df['status'] == 'success'].shape[0]}\n")
        f.write(f"失败计算数量: {df[df['status'] == 'error'].shape[0]}\n\n")
        
        f.write(f"能量统计:\n")
        f.write(f"  最小值: {df['energy'].min()}\n")
        f.write(f"  最大值: {df['energy'].max()}\n")
        f.write(f"  平均值: {df['energy'].mean()}\n")
        f.write(f"  中位数: {df['energy'].median()}\n")
        f.write(f"  标准差: {df['energy'].std()}\n\n")
        
        if 'energy_std' in df.columns:
            f.write(f"VQE重复运行能量波动:\n")
            f.write(f"  平均标准差: {df['energy_std'].mean()}\n")
            f.write(f"  最大标准差: {df['energy_std'].max()}\n\n")
        
        if 'cost_function_evals' in df.columns:
            f.write(f"函数评估次数统计:\n")
            f.write(f"  平均值: {df['cost_function_evals'].mean()}\n")
            f.write(f"  最大值: {df['cost_function_evals'].max()}\n")
            f.write(f"  最小值: {df['cost_function_evals'].min()}\n\n")
        
        if not df['expressibility'].isna().all():
            f.write(f"表达性统计:\n")
            f.write(f"  最小值: {df['expressibility'].min()}\n")
            f.write(f"  最大值: {df['expressibility'].max()}\n")
            f.write(f"  平均值: {df['expressibility'].mean()}\n")
            f.write(f"  中位数: {df['expressibility'].median()}\n")
            f.write(f"  标准差: {df['expressibility'].std()}\n\n")
        
        f.write(f"计算时间统计 (秒):\n")
        f.write(f"  最小值: {df['time_taken'].min()}\n")
        f.write(f"  最大值: {df['time_taken'].max()}\n")
        f.write(f"  平均值: {df['time_taken'].mean()}\n")
        f.write(f"  总计算时间: {df['time_taken'].sum()}\n\n")
        
        f.write(f"回路深度统计:\n")
        f.write(f"  最小值: {df['depth'].min()}\n")
        f.write(f"  最大值: {df['depth'].max()}\n")
        f.write(f"  平均值: {df['depth'].mean()}\n")

def export_results_details(df, output_dir):
    """导出结果的详细CSV文件"""
    # 基本CSV
    basic_csv_path = os.path.join(output_dir, 'results_summary.csv')
    df.to_csv(basic_csv_path, index=False)
    print(f"已导出基本数据到 {basic_csv_path}")
    
    # 尝试导出更详细的CSV，包括所有能量值
    try:
        # 创建扩展数据框架
        extended_df = df.copy()
        
        # 如果有all_energies列，展开为单独的列
        if 'all_energies' in df.columns:
            # 找出最大的能量列表长度
            max_energies = max(df['all_energies'].apply(lambda x: len(x) if isinstance(x, list) else 0))
            
            # 为每个能量创建单独的列
            for i in range(max_energies):
                extended_df[f'energy_{i+1}'] = extended_df['all_energies'].apply(
                    lambda x: x[i] if isinstance(x, list) and i < len(x) else None
                )
            
            # 删除原始列表列
            extended_df = extended_df.drop(columns=['all_energies'])
        
        # 导出详细CSV
        detailed_csv_path = os.path.join(output_dir, 'results_detailed.csv')
        extended_df.to_csv(detailed_csv_path, index=False)
        print(f"已导出详细数据到 {detailed_csv_path}")
    except Exception as e:
        print(f"导出详细数据时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description='分析量子回路计算结果')
    parser.add_argument('--input_dir', type=str, required=True, help='包含批处理结果文件的目录')
    parser.add_argument('--circuit_file', type=str, help='量子回路文件路径（可选）')
    parser.add_argument('--output_dir', type=str, default='analysis_results', help='分析结果输出目录')
    parser.add_argument('--export_csv', action='store_true', help='是否导出CSV文件')
    parser.add_argument('--export_detailed', action='store_true', help='是否导出更详细的CSV文件（包括所有能量值）')
    parser.add_argument('--visualize', action='store_true', help='是否生成可视化图表')
    args = parser.parse_args()
    
    # 加载批处理结果
    results = load_batch_results(args.input_dir)
    if not results:
        print("未找到有效的批处理结果文件，退出。")
        return
    
    # 加载电路（如果提供了文件路径）
    circuits = []
    if args.circuit_file:
        circuits = load_circuits(args.circuit_file)
    
    # 转换为DataFrame
    df = results_to_dataframe(results)
    print(f"成功加载 {len(df)} 条结果记录")
    
    # 显示数据基本统计
    print("\n数据摘要:")
    print(df.describe())
    
    # 统计是否有表达性数据
    has_expressibility = not df['expressibility'].isna().all()
    print(f"\n表达性数据: {'有' if has_expressibility else '无'}")
    
    # 检查是否有重复VQE运行的数据
    has_multiple_vqe = 'energy_std' in df.columns
    print(f"多次VQE运行数据: {'有' if has_multiple_vqe else '无'}")
    
    # 导出CSV
    if args.export_csv or args.export_detailed:
        os.makedirs(args.output_dir, exist_ok=True)
        export_results_details(df, args.output_dir)
    
    # 生成可视化
    if args.visualize:
        print("生成可视化图表...")
        visualize_data(df, args.output_dir)
        print(f"已生成可视化结果到 {args.output_dir}")
    
    print("分析完成!")

if __name__ == "__main__":
    main()