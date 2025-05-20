# 更新config.py添加verbose参数

import argparse
import os
import logging
from datetime import datetime

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

def parse_arguments():
    """解析命令行参数并返回参数对象"""
    parser = argparse.ArgumentParser(description='生成和标记海森堡模型的量子回路')
    
    # 回路生成参数
    circuit_group = parser.add_argument_group('回路生成参数')
    circuit_group.add_argument('--num_circuits', type=int, default=5000, help='生成回路的数量')
    circuit_group.add_argument('--num_qubits', type=int, default=4, help='量子比特数量')
    circuit_group.add_argument('--max_depth', type=int, default=10, help='最大回路深度')
    circuit_group.add_argument('--max_gates', type=int, default=50, help='最大门数量')
    
    # 海森堡模型参数
    heisenberg_group = parser.add_argument_group('海森堡模型参数')
    heisenberg_group.add_argument('--Jx', type=float, default=1.0, help='X方向相互作用强度')
    heisenberg_group.add_argument('--Jy', type=float, default=1.0, help='Y方向相互作用强度')
    heisenberg_group.add_argument('--Jz', type=float, default=1.0, help='Z方向相互作用强度')
    heisenberg_group.add_argument('--hz', type=float, default=1.0, help='Z方向场强')
    
    # 电路生成器参数
    generator_group = parser.add_argument_group('电路生成器参数')
    generator_group.add_argument('--gate_stddev', type=float, default=1.35, help='门选择的标准差')
    generator_group.add_argument('--gate_bias', type=float, default=0.5, help='单量子比特门的偏置')
    
    # 表达性参数
    expr_group = parser.add_argument_group('表达性参数')
    expr_group.add_argument('--calc_expr', action='store_true', help='是否计算表达性')
    expr_group.add_argument('--expr_samples', type=int, default=5000, help='表达性计算的样本数')
    expr_group.add_argument('--expr_bins', type=int, default=75, help='表达性直方图的箱数')
    
    # 标记参数
    labeling_group = parser.add_argument_group('标记参数')
    labeling_group.add_argument('--batch_size', type=int, default=100, help='批处理大小')
    labeling_group.add_argument('--max_workers', type=int, default=None, help='最大工作进程数')
    labeling_group.add_argument('--n_repeat', type=int, default=1, help='每个回路重复VQE的次数')
    labeling_group.add_argument('--verbose', action='store_true', help='显示详细输出（包括VQE优化过程）')
    
    # 输出参数
    output_group = parser.add_argument_group('输出参数')
    output_group.add_argument('--output_dir', type=str, default='output', help='输出目录')
    output_group.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='日志级别')
    output_group.add_argument('--timestamp', action='store_true', help='在输出目录中添加时间戳')
    
    args = parser.parse_args()
    
    # 处理时间戳
    if args.timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(args.output_dir, timestamp)
    
    return args

def get_output_paths(args):
    """根据参数生成输出路径"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成各种输出文件路径
    paths = {
        'circuit_path': os.path.join(args.output_dir, f"circuits_{timestamp}.pkl"),
        'progress_file': os.path.join(args.output_dir, f"progress_{timestamp}.log"),
        'log_file': os.path.join(args.output_dir, f"labeling_{timestamp}.log"),
        'log_main': os.path.join(args.output_dir, f"main_{timestamp}.log")
    }
    
    return paths