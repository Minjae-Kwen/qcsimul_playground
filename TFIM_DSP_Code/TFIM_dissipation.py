import os
import numpy as np
import scipy.linalg as la
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.Floquet import Floquet
import matplotlib.pyplot as plt 

g_per_J_list = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
h_list = [1.61, 1.60, 1.60, 1.65, 1.68, 1.68, 1.69]
theta_over_pi_list = [0.10, 0.10, 0.10, 0.10, 0.10, 0.12, 0.15]
#E_mat_d100 = [0.9340, 0.9363, 0.9195, 0.8941, 0.8901, 0.9088, 0.9077]

# Initial parameters
L, M = 6, 2 # Spin 0-5: System / Spin 6-7: Ancilla
g_per_J = 1.6
d = 160
h = 1.65 
theta = 0.14*np.pi

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
#U_Zanc = la.expm(1j * (np.pi*h/2) * Zanc.toarray())
U_Zanc = la.expm(1j * (np.pi*(h - 0.5)) * Zanc.toarray())

iswap_pairs = [[1.0, 0, L], [1.0, L-1, L+1]]
H_iswap = hamiltonian([["xx", iswap_pairs], ["yy", iswap_pairs]], [], basis=basis, dtype=np.float64)
U_iswap = la.expm(1j * (theta/2) * H_iswap.toarray())

# 3. Randomized initial state
rng = np.random.default_rng(0)
psiS = rng.normal(size=dimS) + 1j*rng.normal(size=dimS)
psiS /= np.linalg.norm(psiS)

psiA0 = np.zeros(dimA, dtype=complex)
psiA0[0] = 1.0
    
psi_full = np.kron(psiS, psiA0)
rho = np.outer(psi_full, psi_full.conj())
rhoA0 = np.outer(psiA0, psiA0.conj())

# 4. Unitary Matrix
U_cycle = U_iswap @ U_Zanc @ U_sys
Udag_cycle = U_cycle.conj().T

# 5. Reference Energy from Hamiltonian Diagonalization
E0 = Hsys.eigsh(k=1, which="SA", return_eigenvectors=False)[0]
print(f"E0 = {E0}")

# Main Loop
reset_freq = 4
E_list = []; d_list = []
for n in range(d):
    if n == 0:
        E = np.real(np.trace(rho @ Hsys.toarray()))
        E_list.append(E/E0)
        d_list.append(n)
    
    rho = U_cycle @ rho @ Udag_cycle

    if (n+1) % reset_freq == 0:
        rho4 = rho.reshape(dimS, dimA, dimS, dimA)             # (s,a,t,a)
        rhoS = np.einsum("sata->st", rho4)  
        rho = np.kron(rhoS, rhoA0)

    E = np.real(np.trace(rho @ Hsys.toarray()))
    E_list.append(E/E0)
    d_list.append(n+1)

print(f"E/E_0 at d = {d_list[-1]}: {E_list[-1]}")

def store_data(npz_path, g_per_J, d_list, E_list):
    
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        
        hit = np.where(np.isclose(data["g_per_J"], g_per_J, atol=1e-12, rtol=0.0))[0]
        if hit.size > 0:
            print(f"Data for g/J = {g_per_J} already exists. Overwriting...")
            idx = int(hit[0])
            d_mat_new = data["d_mat"]; d_mat_new[idx, :] = d_list
            E_mat_new = data["E_mat"]; E_mat_new[idx, :] = E_list
            np.savez(npz_path, g_per_J=data["g_per_J"], d_mat=d_mat_new, E_mat=E_mat_new)
        else:
            g_per_J_new = np.array(data["g_per_J"].tolist() + [g_per_J])
            d_mat_new = np.vstack([data["d_mat"], d_list])
            E_mat_new = np.vstack([data["E_mat"], E_list])
            np.savez(npz_path, g_per_J=g_per_J_new, d_mat=d_mat_new, E_mat=E_mat_new)

    else:
        g_per_J = np.array([g_per_J])
        d_mat = np.array([d_list])
        E_mat = np.array([E_list])
        np.savez(npz_path, g_per_J=g_per_J, d_mat=d_mat, E_mat=E_mat)

def plot_data(npz_path, plot_path):
    # Plotting
    data = np.load(npz_path)
    g_per_J = data["g_per_J"]
    d_mat = data["d_mat"]
    E_mat = data["E_mat"]

    for i, g_val in enumerate(g_per_J):
        plt.plot(d_mat[i], E_mat[i], label=f"g/J={g_val}")
    plt.xlabel("cycle number, d", fontsize=16)
    plt.ylabel(r"$E/E_0$", fontsize=16)
    plt.xlim(left=0.0)
    plt.ylim(bottom=0.0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.7, 0.4))
    plt.savefig(plot_path, bbox_inches="tight")

npz_path = "cooling_log4.npz"; plot_path = "TFIM_dissipative4.png"
store_data(npz_path, g_per_J, d_list, E_list)
plot_data(npz_path, plot_path)
