#!/usr/bin/env python
# 全面修复VQEEnergyLabeler以抑制"Found optimal point"消息

import os
import pickle
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import L_BFGS_B
from expressibility import expressibility 
import io
import logging
from contextlib import redirect_stdout

# 在模块级别设置日志抑制
# 这里我们只暂时降低级别，在函数内部我们会在需要时进一步控制
logging.getLogger('qiskit.algorithms.minimum_eigensolvers.vqe').setLevel(logging.WARNING)
logging.getLogger('qiskit.algorithms.optimizers').setLevel(logging.WARNING)

class VQEEnergyLabeler:
    def __init__(self, 
                 hamiltonian,
                 output_dir,
                 progress_file,
                 log_file,
                 batch_size=512,
                 max_workers=os.cpu_count(),
                 expr_bins=75,
                 expr_samples=5_000,
                 n_repeat=1,
                 expr_qubits=None,
                 calculate_expressibility=True,
                 verbose=False):
        """
        参数：
            hamiltonian: 已生成好的哈密顿量 (SparsePauliOp)
            output_dir: 保存分批结果的目录
            progress_file: 进度日志文件路径
            log_file: 详细结果日志文件路径
            batch_size: 每个批次处理多少个电路
            max_workers: 并行进程数量
            expr_bins: 表达性计算的分箱数
            expr_samples: 表达性计算的样本数
            expr_qubits: 若 None → 运行时自动 len(circ.qubits)
            n_repeat: 重复计算次数
            calculate_expressibility: 是否计算表达性
            verbose: 是否输出详细信息（包括Found optimal point等）
        """
        self.hamiltonian = hamiltonian
        self.output_dir = output_dir
        self.progress_file = progress_file
        self.log_file = log_file
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.expr_bins = expr_bins
        self.expr_samples = expr_samples
        self.expr_qubits = expr_qubits
        self.n_repeat = n_repeat
        self.calculate_expressibility = calculate_expressibility
        self.verbose = verbose

        self.optimizer = L_BFGS_B()
        self.estimator = Estimator()

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # 在初始化时设置日志级别
        self._configure_logging()
            
    def _configure_logging(self):
        """配置日志级别以控制VQE输出"""
        if not self.verbose:
            # 如果不是详细模式，将VQE相关日志设为ERROR级别以抑制大部分消息
            logging.getLogger('qiskit.algorithms.minimum_eigensolvers.vqe').setLevel(logging.ERROR)
            logging.getLogger('qiskit.algorithms.optimizers').setLevel(logging.ERROR)
            # 抑制其他可能的消息源
            logging.getLogger('qiskit.primitives').setLevel(logging.ERROR)
        else:
            # 如果是详细模式，将日志级别设为INFO以显示更多信息
            logging.getLogger('qiskit.algorithms.minimum_eigensolvers.vqe').setLevel(logging.INFO)
            logging.getLogger('qiskit.algorithms.optimizers').setLevel(logging.INFO) 
            logging.getLogger('qiskit.primitives').setLevel(logging.INFO)

    def _log_progress(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.progress_file, 'a') as log:
            log.write(f"{timestamp} - {message}\n")

    def _process_circuit(self, circuit, hamiltonian,
                         global_idx, batch_idx, batch_inner_idx):
        t0 = time.time()
        
        # 根据verbose参数设置日志级别
        if not self.verbose:
            vqe_logger = logging.getLogger('qiskit.algorithms.minimum_eigensolvers.vqe')
            optimizer_logger = logging.getLogger('qiskit.algorithms.optimizers')
            primitives_logger = logging.getLogger('qiskit.primitives')
            
            # 记住原始日志级别
            original_vqe_level = vqe_logger.level
            original_optimizer_level = optimizer_logger.level
            original_primitives_level = primitives_logger.level
            
            # 设置为ERROR级别以抑制大多数消息
            vqe_logger.setLevel(logging.ERROR)
            optimizer_logger.setLevel(logging.ERROR)
            primitives_logger.setLevel(logging.ERROR)

        try:
            # --- 0. 标准化 circuit 对象 ---
            if not isinstance(circuit, QuantumCircuit):
                circuit = QuantumCircuit.from_qasm_str(circuit)

            energies, exprs = [], []
            optimal_params_list = []
            cost_evals_list = []

            # --- 1. 重复 n_repeat 次 ---
            for repeat_idx in range(self.n_repeat):
                # 在详细模式和非详细模式下的处理方式
                if not self.verbose:
                    # 抑制标准输出
                    with redirect_stdout(io.StringIO()):
                        # 运行VQE
                        vqe = VQE(estimator=self.estimator,
                                ansatz=circuit,
                                optimizer=self.optimizer,
                                callback=None)  # 禁用回调以减少输出
                        result = vqe.compute_minimum_eigenvalue(hamiltonian)
                else:
                    # 正常输出模式
                    vqe = VQE(estimator=self.estimator,
                            ansatz=circuit,
                            optimizer=self.optimizer)
                    # 增加详细输出
                    print(f"\n--- 开始运行 VQE (重复 {repeat_idx+1}/{self.n_repeat}) ---")
                    result = vqe.compute_minimum_eigenvalue(hamiltonian)
                    print(f"--- VQE 运行完成 ---\n")
                
                # 从结果中提取能量值
                e = result.eigenvalue.real
                
                # 添加更多详细信息
                optimal_parameters = result.optimal_parameters
                optimal_point = result.optimal_point
                cost_function_evals = result.cost_function_evals
                
                # 保存值
                energies.append(e)
                optimal_params_list.append(optimal_parameters)
                cost_evals_list.append(cost_function_evals)
                
                # 打印能量值和详细信息（如果启用了详细输出）
                if self.verbose:
                    print(f"计算结果: 能量 = {e}")
                    print(f"函数评估次数: {cost_function_evals}")
                
                # 1-B Expressibility (仅在启用时计算)
                if self.calculate_expressibility:
                    with redirect_stdout(io.StringIO()) if not self.verbose else redirect_stdout(None):
                        expr = expressibility(
                            qubits=self.expr_qubits or circuit.num_qubits,
                            circuit=circuit,
                            bins=self.expr_bins,
                            samples=self.expr_samples,
                            show_progress=self.verbose)
                    exprs.append(expr)
                    if self.verbose:
                        print(f"表达性: {expr}")

            # 找到最小能量
            energy_min = min(energies)
            
            # 找到对应最小能量的索引
            min_energy_idx = energies.index(energy_min)
            
            # 找到对应的最优参数和函数评估次数
            optimal_parameters = optimal_params_list[min_energy_idx]
            cost_function_evals = cost_evals_list[min_energy_idx]
            
            # 表达性
            expr_min = min(exprs) if exprs else None
            status = "success"
            err_msg = None

        except Exception as e:
            import traceback
            energy_min = None
            expr_min = None
            status = "error"
            err_msg = f"{str(e)}\n{traceback.format_exc()}"
            optimal_parameters = None
            cost_function_evals = None
            energies = []
            optimal_params_list = []
            cost_evals_list = []
        finally:
            # 恢复原始日志级别
            if not self.verbose:
                vqe_logger.setLevel(original_vqe_level)
                optimizer_logger.setLevel(original_optimizer_level)
                primitives_logger.setLevel(original_primitives_level)

        t1 = time.time()

        # 创建包含更多详细信息的结果
        result = {
            "global_index": global_idx,
            "batch_index": batch_idx,
            "batch_inner_index": batch_inner_idx,
            "optimal_value": energy_min,  # VQE能量值
            "all_energies": energies,     # 所有重复计算的能量值
            "optimal_parameters": optimal_parameters,  # 最优参数
            "cost_function_evals": cost_function_evals,  # 函数评估次数
            "expressibility": expr_min,   # 表达性值
            "status": status,
            "time_taken": t1 - t0,
            "circuit": circuit
        }
        if err_msg is not None:
            result["error_message"] = err_msg
        return result

    def label_energies(self, circuits):
        """
        参数：
            circuits: 已加载到内存中的电路列表
        功能：
            按batch_size分批处理电路，并行运行VQE，日志记录，结果分文件输出
        """

        self._log_progress("Processing started.")
        self._log_progress(f"计算表达性: {'是' if self.calculate_expressibility else '否'}")

        total_circuits = len(circuits)
        total_batches = (total_circuits + self.batch_size - 1) // self.batch_size
        self._log_progress(f"Total circuits: {total_circuits}, total batches: {total_batches}")

        processed_batches = set()
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                for line in f:
                    if "Batch" in line and "processed." in line:
                        parts = line.strip().split()
                        batch_num = int(parts[4])
                        processed_batches.add(batch_num)

        for batch_num in range(1, total_batches + 1):
            if batch_num in processed_batches:
                continue

            start_idx = (batch_num - 1) * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_circuits)
            batch_circuits = circuits[start_idx:end_idx]

            start_time = time.time()
            self._log_progress(f"Processing batch {batch_num}...")

            # 为每个批次创建一个新的进程池，确保资源能被释放
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                tasks = []
                for idx, circuit in enumerate(batch_circuits):
                    global_idx = start_idx + idx
                    tasks.append((circuit, self.hamiltonian, global_idx, batch_num, idx))

                futures = [executor.submit(self._process_circuit, *task) for task in tasks]

                batch_results = []
                for future in as_completed(futures):
                    res = future.result()
                    batch_results.append(res)

                # 批次完成后批量写入日志文件
                with open(self.log_file, 'a') as f_log:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    for result in batch_results:
                        if result["status"] == "success":
                            # 记录更多能量相关信息
                            energy_info = f"Optimal Value: {result['optimal_value']:.6f}"
                            if len(result.get('all_energies', [])) > 1:
                                energy_info += f", Energy Range: [{min(result['all_energies']):.6f} to {max(result['all_energies']):.6f}]"
                            
                            expr_str = f", Expr: {result['expressibility']:.6f}" if result['expressibility'] is not None else ""
                            f_log.write(f"{timestamp} - Batch {result['batch_index']}, "
                                      f"Global {result['global_index']}, Inner {result['batch_inner_index']}: "
                                      f"{energy_info}, Status: success{expr_str}, "
                                      f"Time: {result['time_taken']:.4f}s\n")
                        else:
                            f_log.write(f"{timestamp} - Batch {result['batch_index']}, "
                                      f"Global {result['global_index']}, Inner {result['batch_inner_index']}: "
                                      f"Status: error, Error: {result.get('error_message', 'Unknown')}, "
                                      f"Time: {result['time_taken']:.4f}s\n")

                # 保存当前批次结果到单独文件
                batch_output_file = os.path.join(self.output_dir, f"batch_{batch_num}_results.pkl")
                with open(batch_output_file, 'wb') as f_out:
                    pickle.dump({"results": batch_results}, f_out, protocol=pickle.HIGHEST_PROTOCOL)

            # 批次完全处理完后，显式删除所有相关对象
            del futures
            del batch_results
            del batch_circuits
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
            end_time = time.time()
            batch_time = end_time - start_time
            print(f"Batch {batch_num} processed. Time: {batch_time:.2f} seconds.")
            self._log_progress(f"Batch {batch_num} processed. Time: {batch_time:.2f} seconds.")

        self._log_progress("All batches processed and results saved.")