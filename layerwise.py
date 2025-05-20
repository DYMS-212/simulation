"""
half_layer_heisenberg_generator.py - 基于Half-Layer的海森堡模型量子电路生成器
-------------------------------------------------------------------------------
* 使用half-layer方式生成量子电路（随机选择odd/even比特位）
* 同一half-layer中所有门类型相同
* 跟踪每个量子比特位置上门的历史，避免冗余操作
* 使用元组方式进行电路去重（与gatewise版本一致）
"""

from __future__ import annotations
import numpy as np
import pickle
import logging
from typing import List, Tuple, Set, Optional, Dict, Any
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

# 类型别名定义
Gate = Tuple[int, int, int]      # (gate_type, q1, q2) - 单门 q1=q2
Layer = List[Gate]               # 同一层的多个门
Layers = List[Layer]             # 整个电路
GateMapping = Dict[int, str]     # 门类型ID到Qiskit方法名的映射

# 默认门类型映射
DEFAULT_GATE_MAPPING = {
    1: "rx",    # Rx门
    2: "ry",    # Ry门
    3: "rz",    # Rz门
    4: "rxx",   # Rxx门
    5: "ryy",   # Ryy门
    6: "rzz",   # Rzz门
}

class HalfLayerHeisenbergGenerator:
    """基于Half-Layer的海森堡模型量子电路生成器"""
    
    def __init__(self, 
                 gate_mapping: Optional[GateMapping] = None,
                 num_single_gates: int = 3,
                 logger=None,
                 log_file=None, 
                 log_level=logging.INFO):
        """
        初始化生成器
        
        参数：
            gate_mapping: 门类型ID到Qiskit方法名的映射
            num_single_gates: 单量子比特门的数量
            logger: 日志器实例
            log_file: 日志文件路径
            log_level: 日志级别
        """
        # 设置门类型映射
        self.gate_mapping = gate_mapping or DEFAULT_GATE_MAPPING
        self.num_single_gates = num_single_gates
        
        # 设置日志系统
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger("HalfLayerHeisenbergGenerator")
            self.logger.setLevel(log_level)
            self.logger.propagate = False
            if not self.logger.handlers:
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                
                if log_file is not None:
                    file_handler = logging.FileHandler(log_file, mode='a')
                    file_handler.setLevel(log_level)
                    file_handler.setFormatter(formatter)
                    self.logger.addHandler(file_handler)
                else:
                    console_handler = logging.StreamHandler()
                    console_handler.setLevel(log_level)
                    console_handler.setFormatter(formatter)
                    self.logger.addHandler(console_handler)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """计算softmax以将logits转换为概率分布"""
        exp_x = np.exp(x - np.max(x))  # 防止指数爆炸
        return exp_x / exp_x.sum()
    
    def _generate_gate_probabilities(self, 
                                    gates_count: int, 
                                    gate_stddev: float, 
                                    gate_bias: float,
                                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        生成门类型的概率分布
        
        参数：
            gates_count: 门类型的总数
            gate_stddev: 正态分布的标准差
            gate_bias: 单量子比特门的偏置
            rng: 随机数生成器
        
        返回：
            门类型的概率分布
        """
        rng = rng or np.random.default_rng()
        logits = rng.normal(0, gate_stddev, gates_count)
        # 为单量子比特门添加偏置
        logits[:self.num_single_gates] += gate_bias
        return self._softmax(logits)
    
    def _check_gate_conflicts(self, 
                             gate_type: int, 
                             positions: List[int], 
                             qubit_gate_history: Dict[int, List[int]],
                             num_qubits: int) -> bool:
        """
        检查当前门类型在指定位置是否与历史门冲突
        
        参数：
            gate_type: 当前半层的门类型
            positions: 当前半层可用的位置
            qubit_gate_history: 每个量子比特位置的门类型历史
            num_qubits: 量子比特数量
            
        返回：
            如果存在冲突返回True，否则返回False
        """
        # 检查单量子比特门冲突
        if gate_type <= self.num_single_gates:
            for pos in positions:
                # 检查该位置的历史记录
                if pos in qubit_gate_history:
                    # 如果历史上最后一个门也是相同类型，则冲突
                    if qubit_gate_history[pos] and qubit_gate_history[pos][-1] == gate_type:
                        return True
        
        # 检查双量子比特门冲突
        else:
            for pos in positions:
                next_pos = (pos + 1) % num_qubits
                
                # 检查两个位置的历史记录
                if pos in qubit_gate_history and next_pos in qubit_gate_history:
                    # 如果两个位置最后一个门都是相同类型的双比特门，则冲突
                    # 注意：这是一个简化检查，可能需要更复杂的逻辑来捕获所有情况
                    if (gate_type in qubit_gate_history[pos] and 
                        gate_type in qubit_gate_history[next_pos]):
                        # 进一步检查是否是作为相同的双比特门的一部分
                        # 这里简化处理，只检查最后一个应用的门
                        if (qubit_gate_history[pos][-1] == gate_type and 
                            qubit_gate_history[next_pos][-1] == gate_type):
                            return True
        
        return False
    
    def _update_gate_history(self, 
                           layer: Layer, 
                           qubit_gate_history: Dict[int, List[int]]):
        """
        更新每个量子比特位置的门类型历史
        
        参数：
            layer: 当前层
            qubit_gate_history: 每个量子比特位置的门类型历史
        """
        for gate_type, q1, q2 in layer:
            # 初始化历史记录（如果不存在）
            if q1 not in qubit_gate_history:
                qubit_gate_history[q1] = []
            if q2 not in qubit_gate_history:
                qubit_gate_history[q2] = []
            
            # 更新历史记录
            qubit_gate_history[q1].append(gate_type)
            if q1 != q2:  # 双量子比特门
                qubit_gate_history[q2].append(gate_type)
    
    def _is_adjacent_layer_duplicate(self, gate_type: int, positions: List[int], prev_layer: Layer, num_qubits: int) -> bool:
        """
        检查半层是否与前一层中的门完全相同（位置和种类都相同）
        
        参数：
            gate_type: 当前半层的门类型
            positions: 当前半层可用的位置
            prev_layer: 前一层的门列表
            num_qubits: 量子比特数量
            
        返回：
            如果存在完全相同的门配置返回True，否则返回False
        """
        if not prev_layer:
            return False
            
        # 提取前一层的门类型和位置
        prev_gates = {}
        for prev_gate_type, prev_q1, prev_q2 in prev_layer:
            prev_gates[(prev_q1, prev_q2)] = prev_gate_type
            
        # 检查是否有完全相同的配置（门类型和位置都相同）
        for pos in positions:
            if gate_type <= self.num_single_gates:  # 单量子比特门
                if (pos, pos) in prev_gates and prev_gates[(pos, pos)] == gate_type:
                    return True
            else:  # 双量子比特门
                next_pos = (pos + 1) % num_qubits
                pos_pair = (pos, next_pos) if pos < next_pos else (next_pos, pos)
                if pos_pair in prev_gates and prev_gates[pos_pair] == gate_type:
                    return True
                    
        return False
    
    def generate_half_layer_circuit(self,
                                 num_qubits: int,
                                 max_gates: int,
                                 max_add_count: int,
                                 gate_stddev: float = 1.35,
                                 gate_bias: float = 0.5,
                                 rng: Optional[np.random.Generator] = None) -> Layers:
        """
        基于Half-Layer生成量子电路
        
        参数：
            num_qubits: 量子比特数量
            max_gates: 最大门层数
            max_add_count: 最多添加的门数量
            gate_stddev: 门选择正态分布的标准差
            gate_bias: 单量子比特门的偏置
            rng: 随机数生成器
            
        返回：
            生成的量子电路层结构
        """
        rng = rng or np.random.default_rng()
        gates_count = len(self.gate_mapping)
        
        # 生成门类型的概率分布
        gate_probs = self._generate_gate_probabilities(
            gates_count, gate_stddev, gate_bias, rng)
        
        # 初始化电路结构
        layers = []
        add_count = 0
        step = 0
        
        # 记录前一层，用于相邻层重复检测
        prev_layer = []
        
        # 记录每个量子比特位置的门类型历史
        qubit_gate_history = {}
        
        # 逐层生成电路
        while step < max_gates and add_count < max_add_count:
            current_layer = []
            # 用于跟踪当前层已占用的量子比特
            occupied_qubits = set()
            
            # 随机决定使用奇数位置还是偶数位置
            use_even = rng.choice([True, False])
            positions = list(range(0, num_qubits, 2) if use_even else range(1, num_qubits, 2))
            
            # 尝试选择不冲突的门类型
            max_attempts = 10  # 最大尝试次数
            gate_type = None
            conflict = True
            
            for _ in range(max_attempts):
                # 为整个half-layer随机选择一个门类型
                candidate_gate_type = rng.choice(list(range(1, gates_count + 1)), p=gate_probs)
                
                # 检查是否与前一层重复
                if self._is_adjacent_layer_duplicate(candidate_gate_type, positions, prev_layer, num_qubits):
                    continue
                
                # 检查是否与历史门冲突
                if self._check_gate_conflicts(candidate_gate_type, positions, qubit_gate_history, num_qubits):
                    continue
                
                gate_type = candidate_gate_type
                conflict = False
                break
            
            # 如果所有尝试都冲突，则随机选择一个门类型并继续
            if conflict:
                # 强制换一个门类型，选择冲突最少的
                gate_counts = {}
                for g in range(1, gates_count + 1):
                    gate_counts[g] = 0
                
                # 统计每个门类型在历史记录中的出现次数
                for pos in positions:
                    if pos in qubit_gate_history:
                        for g in qubit_gate_history[pos]:
                            if g in gate_counts:
                                gate_counts[g] += 1
                
                # 选择出现次数最少的门类型
                min_count = float('inf')
                min_gates = []
                for g, count in gate_counts.items():
                    if count < min_count:
                        min_count = count
                        min_gates = [g]
                    elif count == min_count:
                        min_gates.append(g)
                
                # 从最少出现的门类型中随机选择一个
                gate_type = rng.choice(min_gates)
            
            # 将选定的门类型应用到所有可能的位置
            for pos in positions:
                # 如果已经达到最大门数，跳出循环
                if add_count >= max_add_count:
                    break
                    
                # 确定门作用的量子比特
                q1 = pos
                
                # 如果是单量子比特门
                if gate_type <= self.num_single_gates:
                    q2 = q1  # 单门 q1=q2
                    # 检查量子比特是否已被占用
                    if q1 in occupied_qubits:
                        continue
                    
                    # 检查是否与历史门冲突（针对单个位置）
                    if q1 in qubit_gate_history and qubit_gate_history[q1] and qubit_gate_history[q1][-1] == gate_type:
                        continue
                    
                    occupied_qubits.add(q1)
                    current_layer.append((gate_type, q1, q2))
                    add_count += 1
                # 如果是双量子比特门
                else:
                    q2 = (q1 + 1) % num_qubits  # 邻近量子比特
                    # 检查量子比特是否已被占用
                    if q1 in occupied_qubits or q2 in occupied_qubits:
                        continue
                    
                    # 检查是否与历史门冲突（针对单个位置对）
                    if (q1 in qubit_gate_history and q2 in qubit_gate_history and
                        qubit_gate_history[q1] and qubit_gate_history[q2] and
                        qubit_gate_history[q1][-1] == gate_type and qubit_gate_history[q2][-1] == gate_type):
                        continue
                    
                    occupied_qubits.add(q1)
                    occupied_qubits.add(q2)
                    current_layer.append((gate_type, q1, q2))
                    add_count += 1
            
            # 如果当前层添加了门，将其添加到电路中
            if current_layer:
                layers.append(current_layer)
                prev_layer = current_layer.copy()  # 更新前一层记录
                
                # 更新门类型历史
                self._update_gate_history(current_layer, qubit_gate_history)
                
                step += 1
                
        self.logger.info(f"Generated circuit with {add_count} gates in {len(layers)} layers")
        return layers
    
    def layers_to_qiskit(self, layers: Layers) -> QuantumCircuit:
        """
        将层结构转换为Qiskit量子电路
        
        参数：
            layers: 层结构表示的量子电路
            
        返回：
            Qiskit量子电路对象
        """
        # 计算量子比特数量
        num_qubits = 0
        for layer in layers:
            for _, q1, q2 in layer:
                num_qubits = max(num_qubits, q1 + 1, q2 + 1)
        
        # 创建量子电路
        qc = QuantumCircuit(num_qubits)
        param_counter = 1
        
        # 添加门到电路
        for layer in layers:
            for gate_type, q1, q2 in layer:
                # 创建参数
                param = Parameter(f"theta{param_counter}")
                param_counter += 1
                
                # 获取门方法名
                gate_name = self.gate_mapping.get(gate_type)
                if not gate_name:
                    raise ValueError(f"Unknown gate type: {gate_type}")
                
                # 添加门
                if gate_type <= self.num_single_gates:  # 单量子比特门
                    getattr(qc, gate_name)(param, q1)
                else:  # 双量子比特门
                    getattr(qc, gate_name)(param, q1, q2)
        
        return qc
    
    def generate_quantum_circuits(self, 
                                num_circuits: int, 
                                num_qubits: int,
                                max_gates: int,
                                max_add_count: int,
                                gate_stddev: float = 1.35,
                                gate_bias: float = 0.5) -> List[QuantumCircuit]:
        """
        生成多个量子电路
        
        参数：
            num_circuits: 要生成的电路数量
            num_qubits: 量子比特数量
            max_gates: 最大门层数
            max_add_count: 最多添加的门数量
            gate_stddev: 门选择正态分布的标准差
            gate_bias: 单量子比特门的偏置
            
        返回：
            Qiskit量子电路列表
        """
        circuits = []
        unique_circuits = set()  # 用于存储已生成电路的集合（使用元组表示）
        rng = np.random.default_rng()
        
        self.logger.info(f"Starting to generate {num_circuits} quantum circuits...")
        
        while len(circuits) < num_circuits:
            # 生成层结构
            layers = self.generate_half_layer_circuit(
                num_qubits=num_qubits,
                max_gates=max_gates,
                max_add_count=max_add_count,
                gate_stddev=gate_stddev,
                gate_bias=gate_bias,
                rng=rng
            )
            
            # 使用与gatewise版本相同的去重方法 - 将电路展平为元组
            flattened = tuple(gate for layer in layers for gate in layer)
            
            # 检查是否已存在
            if flattened in unique_circuits:
                continue
                
            # 添加到去重集合
            unique_circuits.add(flattened)
            
            # 转换为Qiskit电路
            qc = self.layers_to_qiskit(layers)
            
            # 添加电路ID
            qc.metadata = {"id": len(circuits)}
            circuits.append(qc)
            
            if len(circuits) % 1000 == 0:
                self.logger.info(f"Generated {len(circuits)}/{num_circuits} circuits.")
        
        self.logger.info("All circuits generated successfully.")
        return circuits
    
    def visualize_layers(self, layers: Layers) -> None:
        """
        在终端可视化电路层结构
        
        参数：
            layers: 要可视化的电路层结构
        """
        # 计算量子比特数和层数
        num_qubits = 0
        for layer in layers:
            for _, q1, q2 in layer:
                num_qubits = max(num_qubits, q1 + 1, q2 + 1)
        
        gate_symbol = {
            1: 'Rx', 2: 'Ry', 3: 'Rz',
            4: 'XX', 5: 'YY', 6: 'ZZ'
        }
        
        # 创建可视化网格
        grid = [[' '] * len(layers) for _ in range(num_qubits)]
        
        # 填充网格
        for d, layer in enumerate(layers):
            for g, q1, q2 in layer:
                sym = gate_symbol.get(g, str(g))
                if g <= self.num_single_gates:  # 单量子比特门
                    grid[q1][d] = sym
                else:  # 双量子比特门
                    low, high = sorted((q1, q2))
                    grid[low][d] = sym + '┐'
                    grid[high][d] = sym + '┘'
        
        # 打印可视化结果
        for q, row in enumerate(grid):
            print(f"Q{q}: |" + "".join(f"{c:^4}" for c in row) + "|")
    
    def save_circuits(self, circuits: List[QuantumCircuit], filename: str) -> None:
        """
        保存量子电路到文件
        
        参数：
            circuits: 要保存的量子电路列表
            filename: 输出文件名
        """
        self.logger.info(f"Saving circuits to {filename}...")
        with open(filename, 'wb') as f:
            pickle.dump(circuits, f)
        self.logger.info("Circuits saved successfully.")


