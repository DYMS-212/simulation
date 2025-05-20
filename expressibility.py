"""
Quantum Circuit Expressibility Metrics
--------------------------------------

This module provides tools to calculate expressibility metrics for parameterized quantum circuits.
Expressibility measures how well a parameterized quantum circuit can approximate the Haar measure,
which is useful for evaluating quantum circuit ansatze in hybrid quantum-classical algorithms.

Based on the work by S. Sim, P.D. Johnson and A. Aspuru-Guzik 
"Expressibility and entangling capability of parameterized quantum circuits for hybrid 
quantum-classical algorithms" (Adv. Quantum Technol. 2, 1900070, 2019)
"""

import time
import numpy as np
from typing import Tuple, List, Union, Callable, Optional
from tqdm import tqdm  # For progress bars (install via pip if needed)

from qiskit.quantum_info import state_fidelity, Statevector
from qiskit import QuantumCircuit

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_statevector(circuit: QuantumCircuit) -> np.ndarray:
    """
    Get the statevector representation of a quantum circuit.
    
    Args:
        circuit: QuantumCircuit to convert to statevector
        
    Returns:
        np.ndarray: Complex statevector representation of the circuit
    """
    return Statevector(circuit).data


def p_haar(n_qubits: int, fidelity: float) -> float:
    """
    Calculate the probability density of fidelity values for Haar-random states.
    
    For an n-qubit system, the probability density function is:
    P(F) = (2^n - 1) * (1 - F)^(2^n - 2)
    
    Args:
        n_qubits: Number of qubits in the system
        fidelity: Fidelity value (between 0 and 1)
        
    Returns:
        float: Probability density at the given fidelity
    """
    if fidelity == 1:
        return 0
    else:
        N = 2 ** n_qubits
        return (N - 1) * ((1 - fidelity) ** (N - 2))


def kl_divergence(P: List[float], Q: List[float]) -> float:
    """
    Calculate the Kullback-Leibler divergence between two discrete probability distributions.
    
    KL(P||Q) = Σ P(i) * log(P(i)/Q(i))
    
    Args:
        P: First probability distribution
        Q: Second probability distribution (reference distribution)
        
    Returns:
        float: KL divergence value
        
    Note:
        A small epsilon is added to avoid division by zero or log(0)
    """
    epsilon = 1e-8
    kl_divergence_value = 0.0
    
    for p, q in zip(P, Q):
        if p > 0:  # Only consider non-zero probabilities
            kl_divergence_value += p * np.log((p + epsilon) / (q + epsilon))
    
    return abs(kl_divergence_value)


def bin_fidelities(fidelities: List[float], 
                   n_bins: int = 75) -> Tuple[List[float], List[float]]:
    """
    Create a histogram of fidelity values.
    
    Args:
        fidelities: List of fidelity values
        n_bins: Number of bins for the histogram
        
    Returns:
        Tuple[List[float], List[float]]: Tuple containing:
            - List of bin centers
            - List of probability densities
    """
    hist, bin_edges = np.histogram(fidelities, bins=n_bins, range=(0, 1), density=True)
    bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
    
    # Normalize to create a probability distribution
    hist = hist / sum(hist)
    
    return bin_centers, hist.tolist()


def expressibility(qubits: int, 
                  circuit: QuantumCircuit, 
                  bins: int = 75, 
                  samples: int = 5000, 
                  layer: int = 1, 
                  random_params_fn: Optional[Callable] = None,
                  return_detail: bool = False,
                  show_progress: bool = False,
                  ) -> Union[float, Tuple[List[float], List[float]]]:
    """
    Calculate the expressibility of a parameterized quantum circuit.
    
    Expressibility is measured as the Kullback-Leibler divergence between the
    distribution of fidelities from randomly parameterized circuits and
    the theoretical Haar distribution.
    
    Args:
        qubits: Number of qubits in the circuit
        circuit: Parameterized quantum circuit
        bins: Number of bins for histogram of fidelities
        samples: Number of random circuit pairs to sample
        layer: Number of layers (unused in current implementation)
        random_params_fn: Optional function to generate random parameters
            (Default: Uniform distribution in [0, 2π])
        return_detail: If True, return distributions instead of KL divergence
        show_progress: If True, show a progress bar
        
    Returns:
        If return_detail is False:
            float: Expressibility as KL divergence (lower is better)
        If return_detail is True:
            Tuple[List[float], List[float]]: (Haar distribution, empirical distribution)
    """
    # Check if circuit has parameters
    n_params = len(circuit.parameters)
    if n_params == 0:
        raise ValueError("Circuit must be parameterized")
    
    # Define default random parameter generator if not provided
    if random_params_fn is None:
        random_params_fn = lambda: np.random.uniform(0, 2 * np.pi, size=n_params)
    
    # Define bin width and limits
    unit = 1.0 / bins
    limits = [unit * i for i in range(1, bins + 1)]
    
    # Initialize frequency counts for each bin
    frequencies = np.zeros(bins)
    
    # Create iterator with or without progress bar
    iterator = tqdm(range(samples)) if show_progress else range(samples)
    
    # Sample random circuit pairs and compute fidelities
    for _ in iterator:
        # Generate two sets of random parameters
        random_params_1 = random_params_fn()
        random_params_2 = random_params_fn()
        
        # Create two random circuits
        circuit_1 = circuit.assign_parameters(random_params_1)
        circuit_2 = circuit.assign_parameters(random_params_2)
        
        # Calculate fidelity between the two circuit states
        fidelity = state_fidelity(
            get_statevector(circuit_1),
            get_statevector(circuit_2)
        )
        
        # Bin the fidelity value
        for j in range(bins):
            if fidelity <= limits[j]:
                frequencies[j] += 1
                break
    
    # Normalize frequencies to create a probability distribution
    probabilities = frequencies / samples
    
    # Calculate the theoretical Haar distribution
    bin_centers = [limit - (unit / 2) for limit in limits]
    p_haar_values = [p_haar(qubits, center) / bins for center in bin_centers]
    
    # Return results based on return_detail flag
    if return_detail:
        return p_haar_values, probabilities.tolist()
    else:
        return kl_divergence(probabilities, p_haar_values)


def haar_fidelity_distribution(n_qubits: int, bins: int = 75) -> Tuple[List[float], List[float]]:
    """
    Generate the theoretical Haar random fidelity distribution.
    
    Args:
        n_qubits: Number of qubits
        bins: Number of bins
        
    Returns:
        Tuple[List[float], List[float]]: (bin centers, probabilities)
    """
    unit = 1.0 / bins
    bin_edges = [i * unit for i in range(bins + 1)]
    bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(bins)]
    
    # Calculate Haar probabilities for each bin center
    p_haar_values = [p_haar(n_qubits, center) for center in bin_centers]
    
    # Normalize to create a proper probability distribution
    total = sum(p_haar_values) * unit
    p_haar_normalized = [p / total for p in p_haar_values]
    
    return bin_centers, p_haar_normalized


# Example utility function to plot expressibility distributions
def plot_expressibility_distributions(haar_dist: List[float], 
                                     circuit_dist: List[float], 
                                     bins: int = 75,
                                     circuit_name: str = "Circuit"):
    """
    Plot expressibility distributions for visualization.
    
    Args:
        haar_dist: Haar distribution
        circuit_dist: Circuit distribution
        bins: Number of bins
        circuit_name: Name of the circuit for plot legend
    """
    try:
        import matplotlib.pyplot as plt
        
        bin_centers = [(i + 0.5) / bins for i in range(bins)]
        
        plt.figure(figsize=(10, 6))
        plt.bar(bin_centers, circuit_dist, width=1/bins, alpha=0.6, label=circuit_name)
        plt.plot(bin_centers, haar_dist, 'r-', linewidth=2, label='Haar')
        
        plt.xlabel('Fidelity')
        plt.ylabel('Probability')
        plt.title(f'Expressibility: {circuit_name} vs Haar Distribution')
        plt.legend()
        plt.grid(alpha=0.3)
        
        kl_div = kl_divergence(circuit_dist, haar_dist)
        plt.annotate(f'KL Divergence: {kl_div:.5f}', 
                    xy=(0.7, 0.9), 
                    xycoords='axes fraction', 
                    fontsize=12,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7))
        
        plt.show()
        
    except ImportError:
        print("Matplotlib is required for plotting. Install with 'pip install matplotlib'")


def visualize_states_on_bloch_sphere(circuit, n_samples=5000):
    """
    在带网格线的Bloch球上可视化参数化量子电路生成的量子态。
    
    Args:
        circuit: 参数化量子电路
        n_samples: 要采样的随机参数对数量 (将生成2*n_samples个态)
    """
    # 将态矢量转换为Bloch坐标的函数
    def statevector_to_bloch(sv):
        # 对于单量子比特态 |ψ⟩ = α|0⟩ + β|1⟩
        alpha = sv[0]  # |0⟩的振幅
        beta = sv[1]   # |1⟩的振幅
        
        # 计算Bloch坐标
        x = 2 * np.real(alpha * np.conj(beta))
        y = 2 * np.imag(alpha * np.conj(beta))
        z = np.abs(alpha)**2 - np.abs(beta)**2
        
        return np.array([x, y, z])
    
    # 获取电路中的参数数量
    n_params = len(circuit.parameters)
    
    # 存储Bloch坐标
    bloch_coords = []
    
    # 采样随机参数对
    for _ in range(n_samples):
        # 生成两组随机参数
        random_params_1 = np.random.uniform(0, 2*np.pi, size=n_params)
        random_params_2 = np.random.uniform(0, 2*np.pi, size=n_params)
        
        # 创建带有这些参数的电路
        param_dict_1 = dict(zip(circuit.parameters, random_params_1))
        param_dict_2 = dict(zip(circuit.parameters, random_params_2))
        
        circuit_1 = circuit.assign_parameters(param_dict_1)
        circuit_2 = circuit.assign_parameters(param_dict_2)
        
        # 获取态矢量
        state_1 = Statevector(circuit_1)
        state_2 = Statevector(circuit_2)
        
        # 转换为Bloch坐标
        bloch_1 = statevector_to_bloch(state_1.data)
        bloch_2 = statevector_to_bloch(state_2.data)
        
        # 添加到列表
        bloch_coords.append(bloch_1)
        bloch_coords.append(bloch_2)
    
    # 转换为numpy数组
    bloch_coords = np.array(bloch_coords)
    
    # 准备绘图
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制半透明的Bloch球面
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    x = np.sin(v) * np.cos(u)
    y = np.sin(v) * np.sin(u)
    z = np.cos(v)
    
    # 绘制半透明的Bloch球
    ax.plot_surface(x, y, z, color="lightgray", alpha=0.1, 
                   linewidth=0, antialiased=True)
    
    # 添加球面网格线 - 经线
    for phi in np.linspace(0, 2*np.pi, 12, endpoint=False):
        theta = np.linspace(0, np.pi, 30)
        x_grid = np.sin(theta) * np.cos(phi)
        y_grid = np.sin(theta) * np.sin(phi)
        z_grid = np.cos(theta)
        ax.plot(x_grid, y_grid, z_grid, color='gray', alpha=0.3, linewidth=0.5)
    
    # 添加球面网格线 - 纬线
    for theta in np.linspace(0, np.pi, 6):
        phi = np.linspace(0, 2*np.pi, 60)
        x_grid = np.sin(theta) * np.cos(phi)
        y_grid = np.sin(theta) * np.sin(phi)
        z_grid = np.cos(theta) * np.ones_like(phi)
        ax.plot(x_grid, y_grid, z_grid, color='gray', alpha=0.3, linewidth=0.5)
    
    # 绘制采样的态
    ax.scatter(
        bloch_coords[:, 0], 
        bloch_coords[:, 1], 
        bloch_coords[:, 2], 
        c='blue', 
        alpha=0.5, 
        s=10
    )
    
    # 设置坐标轴范围
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    
    # 关闭背景网格和轴刻度
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # 移除背景面和轴框
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    
    # 添加|0⟩和|1⟩标签
    ax.text(0, 0, 1.15, '|0⟩', fontsize=16, ha='center')
    ax.text(0, 0, -1.15, '|1⟩', fontsize=16, ha='center')
    
    # 添加x和y轴标签
    ax.text(1.15, 0, 0, 'x', fontsize=14)
    ax.text(0, 1.15, 0, 'y', fontsize=14)
    
    # 设置等比例坐标轴
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    plt.show()
    
    return bloch_coords


if __name__ == "__main__":
    # Example usage when running this file directly
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    
    # Create a simple parameterized circuit
    def create_test_circuit(n_qubits=2):
        qc = QuantumCircuit(n_qubits)
        params = []
        
        # First layer
        for i in range(n_qubits):
            param = Parameter(f"θ{i}")
            params.append(param)
            qc.h(i)
            qc.rz(param, i)
        
        # Entangling layer
        for i in range(n_qubits-1):
            qc.cx(i, i+1)
        
        # Second layer
        for i in range(n_qubits):
            param = Parameter(f"φ{i}")
            params.append(param)
            qc.rx(param, i)
        
        return qc
    
    # Create test circuit
    test_circuit = create_test_circuit(2)
    
    print("Calculating expressibility (this may take a while)...")
    start_time = time.time()
    
    # Calculate expressibility with detailed return
    haar_dist, circ_dist = expressibility(
        qubits=2, 
        circuit=test_circuit,
        samples=100,  # Just a small sample for demonstration
        return_detail=True,
        show_progress=True
    )
    
    kl_div = kl_divergence(circ_dist, haar_dist)
    
    print(f"Expressibility (KL Divergence): {kl_div}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    
    # Try to plot if matplotlib is available
    try:
        plot_expressibility_distributions(haar_dist, circ_dist, circuit_name="Test Circuit")
    except Exception as e:
        print(f"Plotting failed: {e}")