#!/usr/bin/env python
# add_expressibility.py - 给已有的能量计算结果添加表达性计算

import os
import pickle
import argparse
import logging
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from expressibility import expressibility
from qiskit import QuantumCircuit

def setup_logging(log_level='INFO', log_file=None):
    """设置日志配置"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'无效的日志级别: {log_level}')
    
    handlers = []
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )

def load_batch_results(directory):
    """加载目录中的所有批处理结果文件"""
    results = {}  # 使用字典以便后续更新
    batch_files = [f for f in os.listdir(directory) if f.startswith('batch_') and f.endswith('_results.pkl')]
    
    logging.info(f"找到 {len(batch_files)} 个批处理结果文件")
    
    for batch_file in sorted(batch_files, key=lambda x: int(x.split('_')[1])):
        batch_num = int(batch_file.split('_')[1])
        file_path = os.path.join(directory, batch_file)
        try:
            with open(file_path, 'rb') as f:
                batch_data = pickle.load(f)
                results[batch_num] = batch_data
            logging.info(f"已加载 {file_path}")
        except Exception as e:
            logging.error(f"加载 {file_path} 失败: {e}")
    
    return results

def save_batch_results(results, directory, preserve_original=True):
    """保存更新后的批处理结果文件"""
    for batch_num, batch_data in results.items():
        file_path = os.path.join(directory, f"batch_{batch_num}_results.pkl")
        
        # 如果需要保留原始文件，则创建备份
        if preserve_original and os.path.exists(file_path):
            backup_path = os.path.join(directory, f"batch_{batch_num}_results_backup.pkl")
            try:
                import shutil
                shutil.copy2(file_path, backup_path)
                logging.info(f"已创建备份文件 {backup_path}")
            except Exception as e:
                logging.error(f"创建备份失败: {e}")
        
        # 保存更新后的结果
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(batch_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f"已保存更新后的结果到 {file_path}")
        except Exception as e:
            logging.error(f"保存结果失败: {e}")

def calculate_expressibility(circuit, qubits, bins, samples, show_progress=False):
    """计算单个电路的表达性"""
    try:
        # 确保circuit是QuantumCircuit对象
        if not isinstance(circuit, QuantumCircuit):
            circuit = QuantumCircuit.from_qasm_str(circuit)
        
        # 计算表达性
        expr = expressibility(
            qubits=qubits,
            circuit=circuit,
            bins=bins,
            samples=samples,
            show_progress=show_progress
        )
        return expr
    except Exception as e:
        logging.error(f"表达性计算失败: {e}")
        return None

def process_result(result, qubits, bins, samples):
    """处理单个结果项，添加表达性计算"""
    # 如果已经有表达性值，或者状态不是成功，则跳过
    if result['expressibility'] is not None or result['status'] != 'success':
        return result
    
    start_time = time.time()
    
    # 计算表达性
    expr = calculate_expressibility(
        circuit=result['circuit'],
        qubits=qubits,
        bins=bins,
        samples=samples
    )
    
    # 更新结果
    result['expressibility'] = expr
    result['expr_time_taken'] = time.time() - start_time
    
    return result

def main():
    parser = argparse.ArgumentParser(description='给已有的能量计算结果添加表达性计算')
    parser.add_argument('--input_dir', type=str, required=True, help='包含批处理结果文件的目录')
    parser.add_argument('--output_dir', type=str, help='结果输出目录，默认与输入目录相同')
    parser.add_argument('--expr_samples', type=int, default=5000, help='表达性计算的样本数')
    parser.add_argument('--expr_bins', type=int, default=75, help='表达性直方图的箱数')
    parser.add_argument('--qubits', type=int, help='量子比特数量，默认从回路中自动获取')
    parser.add_argument('--max_workers', type=int, default=None, help='最大工作进程数')
    parser.add_argument('--batch', type=int, help='只处理指定批次号的结果')
    parser.add_argument('--preserve', action='store_true', help='保留原始结果文件的备份')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='日志级别')
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = args.output_dir or args.input_dir
    
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"add_expressibility_{timestamp}.log")
    setup_logging(args.log_level, log_file)
    
    # 加载批处理结果
    batch_results = load_batch_results(args.input_dir)
    if not batch_results:
        logging.error("未找到有效的批处理结果文件，退出。")
        return
    
    # 如果指定了批次，只处理该批次
    if args.batch is not None:
        if args.batch in batch_results:
            batch_results = {args.batch: batch_results[args.batch]}
            logging.info(f"只处理批次 {args.batch}")
        else:
            logging.error(f"找不到批次 {args.batch}，退出。")
            return
    
    # 统计需要计算表达性的结果数量
    total_results = 0
    need_expr_count = 0
    for batch_num, batch_data in batch_results.items():
        for result in batch_data['results']:
            total_results += 1
            if result['expressibility'] is None and result['status'] == 'success':
                need_expr_count += 1
    
    logging.info(f"总结果数量: {total_results}, 需要计算表达性的数量: {need_expr_count}")
    
    if need_expr_count == 0:
        logging.info("所有结果都已经计算表达性或计算失败，无需重新计算。")
        return
    
    # 开始计算表达性
    logging.info(f"开始计算表达性，使用 {args.max_workers or os.cpu_count()} 个进程...")
    start_time = time.time()
    
    # 使用进程池并行计算
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        for batch_num, batch_data in batch_results.items():
            logging.info(f"处理批次 {batch_num}...")
            futures = []
            
            # 提交任务
            for i, result in enumerate(batch_data['results']):
                if result['expressibility'] is None and result['status'] == 'success':
                    # 确定要使用的量子比特数
                    qubits = args.qubits
                    if qubits is None and isinstance(result['circuit'], QuantumCircuit):
                        qubits = result['circuit'].num_qubits
                    
                    if qubits is None:
                        logging.warning(f"批次 {batch_num}, 结果 {i}: 无法确定量子比特数量，跳过")
                        continue
                    
                    # 提交计算任务
                    future = executor.submit(
                        process_result,
                        result.copy(),  # 创建副本以避免引用问题
                        qubits,
                        args.expr_bins,
                        args.expr_samples
                    )
                    futures.append((i, future))
            
            # 收集结果
            for i, future in futures:
                try:
                    updated_result = future.result()
                    batch_data['results'][i] = updated_result
                    logging.info(f"批次 {batch_num}, 结果 {i}: 表达性计算完成, 值 = {updated_result['expressibility']}")
                except Exception as e:
                    logging.error(f"批次 {batch_num}, 结果 {i}: 处理失败: {e}")
    
    # 计算耗时
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"表达性计算完成，总耗时: {elapsed_time:.2f} 秒")
    
    # 保存更新后的结果
    save_batch_results(batch_results, output_dir, args.preserve)
    logging.info(f"所有更新后的结果已保存到 {output_dir}")

if __name__ == "__main__":
    main()
