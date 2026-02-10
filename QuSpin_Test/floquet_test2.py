import numpy as np
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.Floquet import Floquet
import matplotlib.pyplot as plt 

# Initial parameters
L = 4
J = 1.0
h = np.sqrt(2)

# Coupling lists
h_field = [[-h, i] for i in range(L)]
J_zz = [[-J, i, (i + 1) % L] for i in range(L)]  # PBC

# Floquet step durations
switch_n = 5
tau_zz = 0.5; tau_x  = 0.5
dt_zz = tau_zz / switch_n
dt_x  = tau_x  / switch_n
T = tau_zz + tau_x

E_Ham = {}; E_Floq = {}
V_Ham = {}; V_Floq = {}

# Main Loop
for zblock in [-1, +1]:
    basis = spin_basis_1d(L=L, zblock=zblock)

    # Hamiltonians
    H_overall = hamiltonian([["zz", J_zz], ["x", h_field]], [], basis=basis, dtype=np.float64)
    Hzz = hamiltonian([["zz", J_zz]], [], basis=basis, dtype=np.float64)
    Hx  = hamiltonian([["x",  h_field]], [], basis=basis, dtype=np.float64)

    # Hamiltonian Diagonalization
    EH, VH = H_overall.eigh()
    EH_exp = -np.angle(np.exp(-1j * EH * T)) / T
    idxH = np.argsort(EH_exp)
    E_Ham[zblock] = EH_exp[idxH]
    V_Ham[zblock] = VH[:, idxH]

    # Floquet
    evo_dict = {
        "H_list": [2*Hzz, 2*Hx] * switch_n,  # step Hamiltonians
        "dt_list": np.array([dt_zz, dt_x]*switch_n, dtype=np.float64)
    }

    Floq = Floquet(evo_dict, UF=True, VF=True, thetaF=True)

    idxF = np.argsort(Floq.EF)
    E_Floq[zblock] = Floq.EF[idxF]
    V_Floq[zblock] = Floq.VF[:, idxF]

    M = np.abs(V_Ham[zblock].conj().T @ V_Floq[zblock])**2
    print(f"For zblock={zblock}, diagonal elements of Overlap are: {np.diag(M)}")  

print(f"For zblock={zblock}, \nE_Ham: {E_Ham[zblock]}, \nE_Floq: {E_Floq[zblock]}")

VHsub = V_Ham[1][:, 2:6]
VFsub = V_Floq[1][:, 2:6]
S = VHsub.conj().T @ VFsub              # (4,4)
sv = np.linalg.svd(S, compute_uv=False) # singular values
print("\nSingular values:", sv)