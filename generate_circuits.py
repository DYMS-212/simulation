# 修改generate_circuits.py脚本以使用改进的VQEEnergyLabeler

#!/usr/bin/env python
# 主调用脚本：generate_circuits.py

import time
import pickle
import logging

# 导入配置模块
from config import parse_arguments, setup_logging, get_output_paths

# 导入功能模块
from Heisenberg import Heisenberg
from layerwise import HalfLayerHeisenbergGenerator
from energy_label import VQEEnergyLabeler  # 导入修改后的VQEEnergyLabeler

def main():
    # 解析命令行参数
    args = parse_arguments()
    
    # 获取输出路径
    paths = get_output_paths(args)
    
    # 设置日志
    setup_logging(args.log_level, paths['log_main'])
    
    try:
        # 步骤1: 创建海森堡哈密顿量
        logging.info(f"为{args.num_qubits}个量子比特创建海森堡哈密顿量...")
        heisenberg = Heisenberg(
            size=args.num_qubits,
            Jx=args.Jx,
            Jy=args.Jy,
            Jz=args.Jz,
            hz=args.hz
        )
        hamiltonian, energy = heisenberg.get_hamiltonian_and_energy()
        logging.info(f"哈密顿量创建成功. 精确基态能量: {energy}")
        
        # 步骤2: 生成量子回路
        logging.info(f"生成{args.num_circuits}个量子回路...")
        generator = HalfLayerHeisenbergGenerator(
            log_level=getattr(logging, args.log_level)
        )
        circuits = generator.generate_quantum_circuits(
            num_circuits=args.num_circuits,
            num_qubits=args.num_qubits,
            max_gates=args.max_depth,
            max_add_count=args.max_gates,
            gate_stddev=args.gate_stddev,
            gate_bias=args.gate_bias
        )
        logging.info(f"已生成{len(circuits)}个回路.")
        
        # 保存回路
        with open(paths['circuit_path'], 'wb') as f:
            pickle.dump(circuits, f)
        logging.info(f"已将回路保存到{paths['circuit_path']}")
        
        # 步骤3: 标记回路
        logging.info(f"{'计算' if args.calc_expr else '不计算'}表达性...")
        
        # 添加新的命令行参数到config.py中
        verbose = getattr(args, 'verbose', False)  # 如果参数不存在，默认为False
        
        # 初始化标记器，添加详细输出控制参数
        labeler = VQEEnergyLabeler(
            hamiltonian=hamiltonian,
            output_dir=args.output_dir,
            progress_file=paths['progress_file'],
            log_file=paths['log_file'],
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            expr_bins=args.expr_bins,
            expr_samples=args.expr_samples,
            n_repeat=args.n_repeat,
            expr_qubits=args.num_qubits,
            calculate_expressibility=args.calc_expr,
            verbose=verbose  # 传递详细输出控制参数
        )
        
        # 标记回路
        labeler.label_energies(circuits)
        logging.info("标记完成.")
        
        logging.info(f"所有操作都成功完成。结果保存在{args.output_dir}")
    
    except Exception as e:
        logging.exception(f"发生错误: {e}")
        raise

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"总执行时间: {end_time - start_time:.2f} 秒")