from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import NumPyMinimumEigensolver


class Heisenberg:
    def __init__(self, size=8, Jx=1.0, Jy=1.0, Jz=1.0, hz=1.0):
        """
        初始化海森堡模型的实例。

        参数:
        size (int): 系统的大小。
        Jx, Jy, Jz (float): X, Y, Z方向的相互作用强度。
        hz (float): 单体项Z的强度。
        """
        self.size = size
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.hz = hz
        self.H_list = []

    def generate_hamiltonian(self):
        """
        生成海森堡模型的哈密顿量。

        返回:
        SparsePauliOp: 系统的哈密顿量。
        """
        self.H_list = []
        # 构造相互作用项：X_i X_{i+1}, Y_i Y_{i+1}, Z_i Z_{i+1}
        for i in range(self.size):
            for interaction, coefficient in [('X', self.Jx), ('Y', self.Jy), ('Z', self.Jz)]:
                term = ''.join(interaction if k == i or k == (i + 1) %
                               self.size else 'I' for k in range(self.size))
                self.H_list.append((term, coefficient))

        # 构造单体项：Z_i
        for i in range(self.size):
            term = ''.join('Z' if k == i else 'I' for k in range(self.size))
            self.H_list.append((term, self.hz))

        self.Hamiltonian = SparsePauliOp.from_list(self.H_list)
        return self.Hamiltonian

    def compute_energy(self):
        """
        计算海森堡模型的基态能量。

        返回:
        float: 系统的基态能量。
        """
        numpy_solver = NumPyMinimumEigensolver()
        result = numpy_solver.compute_minimum_eigenvalue(self.Hamiltonian)
        return result.eigenvalue.real

    def get_hamiltonian_and_energy(self):
        """
        获取海森堡模型的哈密顿量和基态能量。

        返回:
        tuple: 包含哈密顿量 (SparsePauliOp) 和基态能量 (float) 的元组。
        """
        hamiltonian = self.generate_hamiltonian()
        energy = self.compute_energy()
        return hamiltonian, energy


# 使用示例
if __name__ == "__main__":
    heisenberg = Heisenberg(size=8, Jx=1.0, Jy=1.0, Jz=1.0, hz=1.0)
    hamiltonian, energy = heisenberg.get_hamiltonian_and_energy()
    print("Hamiltonian:", hamiltonian)
    print("Ground state energy:", energy)
