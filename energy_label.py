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

class VQEEnergyLabeler:
    def __init__(self, 
                 hamiltonian,
                 output_dir,
                 progress_file,
                 log_file,
                 batch_size=512,
                 max_workers=os.cpu_count(),
                 # ↓ 新增 4 个可调参数
                 expr_bins       = 75,       # ↓ 新增 3 个可调参数
                 expr_samples    = 5_000,
                 n_repeat     = 1, # 重复次数
                 expr_qubits     = None):    # 默认为电路 qubit 数
        """
        参数：
            hamiltonian: 已生成好的哈密顿量 (SparsePauliOp)
            output_dir: 保存分批结果的目录
            progress_file: 进度日志文件路径, 用于记录处理进度,后续可用于断点续传
            log_file: 详细结果日志文件路径
            batch_size: 每个批次处理多少个电路
            max_workers: 并行进程数量
        """
        self.hamiltonian = hamiltonian
        self.output_dir = output_dir
        self.progress_file = progress_file
        self.log_file = log_file
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.expr_bins    = expr_bins
        self.expr_samples = expr_samples
        self.expr_qubits  = expr_qubits      # 若 None → 运行时自动 len(circ.qubits)
        self.n_repeat     = n_repeat         # 重复次数

        self.optimizer = L_BFGS_B()
        self.estimator = Estimator()

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _log_progress(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.progress_file, 'a') as log:
            log.write(f"{timestamp} - {message}\n")

    def _process_circuit(self, circuit, hamiltonian,
                         global_idx, batch_idx, batch_inner_idx):
        t0 = time.time()

        try:
            # --- 0. 标准化 circuit 对象 ---
            if not isinstance(circuit, QuantumCircuit):
                circuit = QuantumCircuit.from_qasm_str(circuit)

            energies, exprs = [], []

            # --- 1. 重复 n_repeat 次 ---
            for _ in range(self.n_repeat):
                # 1-A VQE
                vqe = VQE(estimator=self.estimator,
                          ansatz=circuit,
                          optimizer=self.optimizer)
                e = vqe.compute_minimum_eigenvalue(hamiltonian).eigenvalue.real
                energies.append(e)

                # 1-B Expressibility
                expr = expressibility(
                    qubits   = self.expr_qubits or circuit.num_qubits,
                    circuit  = circuit,
                    bins     = self.expr_bins,
                    samples  = self.expr_samples,
                    show_progress=False)
                exprs.append(expr)

            energy_min = min(energies)
            expr_min   = min(exprs)
            status     = "success"
            err_msg    = None

        except Exception as e:
            energy_min = None
            expr_min   = None
            status     = "error"
            err_msg    = str(e)

        t1 = time.time()

        result = {
            "global_index"     : global_idx,
            "batch_index"      : batch_idx,
            "batch_inner_index": batch_inner_idx,
            "optimal_value"    : energy_min,
            "expressibility"   : expr_min,
            "status"           : status,
            "time_taken"       : t1 - t0,
            "circuit"          : circuit
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
            加入了断点续传功能
            加入内存管理
        """

        self._log_progress("Processing started.")

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
                            f_log.write(f"{timestamp} - Batch {result['batch_index']}, "
                                        f"Global {result['global_index']}, Inner {result['batch_inner_index']}: "
                                        f"Optimal Value: {result['optimal_value']}, Status: success, "
                                        f"Expr: {result['expressibility']:.6f}, "
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
            
            # # 如果使用了CUDA，可能需要清理GPU内存
            # try:
            #     import torch
            #     if torch.cuda.is_available():
            #         torch.cuda.empty_cache()
            # except ImportError:
            #     pass
            
            end_time = time.time()
            batch_time = end_time - start_time
            print(f"Batch {batch_num} processed. Time: {batch_time:.2f} seconds.")
            self._log_progress(f"Batch {batch_num} processed. Time: {batch_time:.2f} seconds.")

        self._log_progress("All batches processed and results saved.")
