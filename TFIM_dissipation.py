import numpy as np
import scipy.linalg as la
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.Floquet import Floquet
import matplotlib.pyplot as plt 

# Initial parameters
L, M = 6, 2
g_per_J = 1.6
d = 5000
h = 1.65   
theta = 0.11*np.pi

Ltot = L + M
J = 0.2
g = g_per_J * J
t_unit = np.pi/2
T = t_unit * d
dimS = 2**L
dimA = 2**M

# 1. Prep for Bases, Hamiltonian, Unitary Matrix, etc
basis = spin_basis_1d(L=Ltot)

h_z = [[-g, i] for i in range(L)]
h_xx = [[J, i, i + 1] for i in range(L-1)]

Hz  = hamiltonian([["z",  h_z]], [], basis=basis, dtype=np.float64)
Hxx = hamiltonian([["xx", h_xx]], [], basis=basis, dtype=np.float64)
Hsys = Hz + Hxx  # System Hamiltonian in matrix form
U_sys = la.expm(-1j * (np.pi/2) * Hz.toarray()) @ la.expm(-1j * (np.pi/2) * Hxx.toarray())
'''
evo_dict = {
    "H_list": [Hxx, Hz],  # step Hamiltonians
    "dt_list": np.array([t_unit, t_unit], dtype=np.float64),
    "T": t_unit # !!! OR t_unit * 2
}

Floq = Floquet(evo_dict, UF=True, VF=True, thetaF=True)
E_floq = Floq.EF
V_floq = Floq.VF
U_sys = np.kron(np.eye(dimA, dtype=complex), Floq.UF)  # Ancilla identity x System Floquet unitary
'''
# 2. Ancilla Unitary Operators
z_anc_list = [[1.0, i+L] for i in range(M)]
Zanc = hamiltonian([["z", z_anc_list]], [], basis=basis, dtype=np.float64)
U_Zanc = la.expm(-1j * (np.pi*h/2) * Zanc.toarray())

iswap_pairs = [[1.0, 0, L], [1.0, L-1, L+1]]
H_iswap = hamiltonian([["xx", iswap_pairs], ["yy", iswap_pairs]], [], basis=basis, dtype=np.float64)
U_iswap = la.expm(1j * (theta/2) * H_iswap.toarray())

# 3. Randomized initial state
rng = np.random.default_rng(0)
psiS = rng.normal(size=dimS) + 1j*rng.normal(size=dimS)
psiS /= np.linalg.norm(psiS)

psiA0 = np.zeros(dimA, dtype=complex)
psiA0[0] = 1.0
    
psi_full = np.kron(psiA0, psiS)
rho = np.outer(psi_full, psi_full.conj())
rhoA0 = np.outer(psiA0, psiA0.conj())

# 4. Unitary Matrix
U_cycle = U_Zanc @ U_iswap @ U_sys
Udag_cycle = U_cycle.conj().T

# 5. Reference Energy from Hamiltonian Diagonalization
E0 = Hsys.eigsh(k=1, which="SA", return_eigenvectors=False)[0]
print(f"E0 = {E0}")

# Main Loop
reset_freq = 4
E_list = []; d_list = []
for n in range(d):
    rho = U_cycle @ rho @ Udag_cycle

    if (n+1) % reset_freq == 0:
        rho4 = rho.reshape(dimA, dimS, dimA, dimS)
        rhoS = np.einsum("asat->st", rho4)
        rho = np.kron(rhoA0, rhoS)

        E = np.real(np.trace(rho @ Hsys.toarray()))
        E_list.append(E/E0)
        d_list.append(n)

plt.plot(d_list, E_list, marker="o")
plt.xlabel("cycle number, d", fontsize=16)
plt.ylabel("E/E0", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
#plt.legend(fontsize=16)
plt.grid()
plt.tight_layout()
plt.savefig(f"TFIM_dissipative_gJ={g_per_J}_h={h}_theta={theta}.png", bbox_inches="tight")

'''

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
'''