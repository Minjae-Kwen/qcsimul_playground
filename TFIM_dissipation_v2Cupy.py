import os
import numpy as np
import cupy as cp
import scipy.linalg as la
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.Floquet import Floquet
import matplotlib.pyplot as plt
from tqdm import tqdm

# Initial parameters
L, M = 6, 2 # Spin 0-5: System / Spin 6-7: Ancilla
dimS = 2**L; dimA = 2**M
d = 100
t_unit = np.pi/2
T = t_unit * d

g_per_J = 0.4
J = 0.2
g = g_per_J * J

reset_freq = 4

#h = (1.65/1.887) * np.sqrt(1.0 + g_per_J**2)
#theta = 0.11*np.pi
h_list = np.linspace(1.0, 3.0, 21)
theta_over_pi_list = np.linspace(0.05, 0.35, 31)

# Method Definition
def prep_tEvolution(L, M, g, J, h, theta):
    Ltot = L + M
    #dimS = 2**L; dimA = 2**M

    # 1. Prep for Bases, Hamiltonian, Unitary Matrix, etc
    basis = spin_basis_1d(L=Ltot)

    h_z = [[-g, i] for i in range(L)]
    h_xx = [[J, i, i + 1] for i in range(L-1)]

    Hz  = hamiltonian([["z",  h_z]], [], basis=basis, dtype=np.float64)
    Hxx = hamiltonian([["xx", h_xx]], [], basis=basis, dtype=np.float64)
    Hsys = Hz + Hxx  # System Hamiltonian in matrix form
    U_sys = la.expm(-1j * (np.pi/2) * Hz.toarray()) @ la.expm(-1j * (np.pi/2) * Hxx.toarray())

    # 2. Ancilla Unitary Operators
    z_anc_list = [[1.0, i+L] for i in range(M)]
    Zanc = hamiltonian([["z", z_anc_list]], [], basis=basis, dtype=np.float64)
    U_Zanc = la.expm(1j * (np.pi*h/2) * Zanc.toarray())

    iswap_pairs = [[1.0, 0, L], [1.0, L-1, L+1]]
    H_iswap = hamiltonian([["xx", iswap_pairs], ["yy", iswap_pairs]], [], basis=basis, dtype=np.float64)
    U_iswap = la.expm(1j * (theta/2) * H_iswap.toarray())

    U_cycle = U_Zanc @ U_iswap @ U_sys

    return U_cycle, Hsys

def prep_initial_state(L, M):
    dimS = 2**L; dimA = 2**M

    # 3. Randomized initial state
    rng = np.random.default_rng(0)
    psiS = rng.normal(size=dimS) + 1j*rng.normal(size=dimS)
    psiS /= np.linalg.norm(psiS)

    psiA0 = np.zeros(dimA, dtype=complex)
    psiA0[0] = 1.0
        
    psi_full = np.kron(psiS, psiA0)
    rho = np.outer(psi_full, psi_full.conj())
    rhoA0 = np.outer(psiA0, psiA0.conj())

    return rho, rhoA0

def auxiliary_method(Hsys, L, M, t_unit):
    # A. Reference Energy from Hamiltonian Diagonalization
    E0 = Hsys.eigsh(k=1, which="SA", return_eigenvectors=False)[0]
    #print(f"E0 = {E0}")

    # B. Floquet wavefunctions
    basisS = spin_basis_1d(L=L)
    HzS  = hamiltonian([["z",  [[-g, i] for i in range(L)] ]], [], basis=basisS, dtype=np.float64)
    HxxS = hamiltonian([["xx", [[J, i, i + 1] for i in range(L-1)] ]], [], basis=basisS, dtype=np.float64)
    evo_dict = {
        "H_list": [HxxS, HzS],  # step Hamiltonians
        "dt_list": np.array([t_unit, t_unit], dtype=np.float64)
    }

    Floq = Floquet(evo_dict, UF=True, VF=True, thetaF=True)
    idx_gs = np.argmin(Floq.EF)
    psi0_sys = Floq.VF[:, idx_gs] 
    psi0_anc = np.zeros(2**M, dtype=complex); psi0_anc[0] = 1.0
    psi0 = np.kron(psi0_sys, psi0_anc)

    return E0, psi0

def run_grid_cupy(h_list, theta_over_pi_list,
                  L, M, g, J, d, reset_freq, t_unit,
                  batch_size=None, use_complex64=True):

    # ---- sizes ----
    dimS = 2**L
    dimA = 2**M
    D = dimS * dimA

    n_th = len(theta_over_pi_list)
    n_h  = len(h_list)
    P = n_th * n_h

    ctype = cp.complex64 if use_complex64 else cp.complex128
    rtype = cp.float32   if use_complex64 else cp.float64

    # ---- (1) CPU에서 U_cycle, Hsys, E0를 만들어서 쌓기 (여긴 아직 CPU 루프) ----
    U_list  = []
    H_list  = []
    E0_list = []

    for theta_over_pi in theta_over_pi_list:
        theta = theta_over_pi * np.pi
        for h in h_list:
            U_cycle, Hsys = prep_tEvolution(L, M, g, J, h, theta)
            
            U_np = np.asarray(U_cycle, dtype=np.complex128)
            H_np = np.asarray(Hsys.toarray(), dtype=np.float64)

            E0, _ = auxiliary_method(Hsys, L, M, t_unit)

            U_list.append(U_np)
            H_list.append(H_np)
            E0_list.append(E0)

    U_np  = np.stack(U_list, axis=0)                # (P, D, D)
    H_np  = np.stack(H_list, axis=0)                # (P, D, D)
    E0_np = np.array(E0_list, dtype=np.float64)     # (P,)

    # ---- (2) 초기 상태 (CPU -> GPU) ----
    rho0_np, rhoA0_np = prep_initial_state(L, M)    # 네 함수
    rho0_cp  = cp.asarray(rho0_np, dtype=ctype)     # (D, D)
    rhoA0_cp = cp.asarray(rhoA0_np, dtype=ctype)    # (dimA, dimA)

    # ---- (3) 배치 계산 (메모리 아끼려면 chunk 처리 가능) ----
    if batch_size is None:
        batch_size = P  # 한 번에

    out = np.empty(P, dtype=np.float64)

    for s in range(0, P, batch_size):
        e = min(P, s + batch_size)
        B = e - s

        U  = cp.asarray(U_np[s:e], dtype=ctype)                # (B, D, D)
        H  = cp.asarray(H_np[s:e], dtype=ctype)                # (B, D, D) -> complex로 맞춰둠
        E0 = cp.asarray(E0_np[s:e], dtype=cp.float64)          # (B,)

        Udag = cp.swapaxes(U.conj(), -1, -2)                   # (B, D, D)

        # rho 배치: 동일 초기 rho0를 B개 복제
        rho = cp.broadcast_to(rho0_cp, (B, D, D)).copy()

        for n in range(d):
            # rho = U rho U†  (batched matmul)
            rho = cp.matmul(U, cp.matmul(rho, Udag))

            if (n + 1) % reset_freq == 0:
                # S ⊗ A 순서 가정: (s,a,t,a)
                rho4 = rho.reshape(B, dimS, dimA, dimS, dimA)
                rhoS = cp.einsum("bsata->bst", rho4)  # trace over ancilla 'a'

                # rho <- rhoS ⊗ rhoA0  (배치 kron: cp.kron은 배치 안 됨 -> broadcasting으로 구현)
                rho = (rhoS[:, :, None, :, None] * rhoA0_cp[None, None, :, None, :]).reshape(B, D, D)

        # E = Tr(rho H)
        # batched trace: sum_{i,j} rho_{ij} H_{ji}
        E = cp.real(cp.einsum("bij,bji->b", rho, H))
        val = (E / E0).get()   # GPU -> CPU
        out[s:e] = val

    # ---- (4) (n_th, n_h)로 reshape ----
    E_E0_mat = out.reshape(n_th, n_h)

    idx = np.nanargmax(E_E0_mat)
    i,j = np.unravel_index(idx, E_E0_mat.shape)
    Emax = E_E0_mat[i, j]
    theta_at_max = theta_over_pi_list[i] * np.pi
    h_at_max = h_list[j]

    return E_E0_mat, Emax, theta_at_max, h_at_max
 
E_E0_mat = np.zeros((len(theta_over_pi_list), len(h_list)), dtype=np.float64)
E_E0_mat, Emax, theta_at_max, h_at_max = run_grid_cupy(h_list, theta_over_pi_list, L, M, g, J, d, reset_freq, t_unit)
print(f"Max E/E0 = {Emax:.4f} at theta/pi = {theta_at_max/np.pi:.3f}, h = {h_at_max:.3f}")

############################################

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
        plt.plot(d_mat[i], E_mat[i], marker="o", label=f"g/J={g_val}")
    plt.xlabel("cycle number, d", fontsize=16)
    plt.ylabel(r"$E/E_0$", fontsize=16)
    plt.ylim(bottom=0.0)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    plt.savefig(plot_path, bbox_inches="tight")

def plot_heatmap(theta_over_pi_list, h_list, E_E0_mat, plot_path):
    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=150)
    im = ax.imshow(
        E_E0_mat,
        origin="lower",
        extent=[h_list[0], h_list[-1], theta_over_pi_list[0], theta_over_pi_list[-1]],
        aspect="auto",
        interpolation="bilinear", 
        cmap = "viridis_r",
        vmin=0.0, vmax=1.0
    )

    ax.set_xlabel("h")
    ax.set_ylabel(r"$\theta/\pi$ (rad)")

    ax.text(
        0.03, 0.97,
        f" J = {J:.2f},\ng = {g:.2f},\nd = {d}",
        transform=ax.transAxes,
        ha="left", va="top",
        color="white", fontsize=9, 
    )

    ax.text(
        0.55, 0.35,
        f"E = {Emax:.4f},\nθ/π = {theta_at_max/np.pi:.2f},\nh = {h_at_max:.2f}",
        transform=ax.transAxes,
        ha="left", va="top",
        color="white", fontsize=7, 
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$E/E_0$")

    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")

#npz_path = "cooling_log3.npz"; plot_path = "TFIM_dissipative3.png"
#store_data(npz_path, g_per_J, d_list, E_list)
#plot_data(npz_path, plot_path)
plot_path = f"TFIM_h_theta_colormap(gJ={g_per_J}).png"
plot_heatmap(theta_over_pi_list, h_list, E_E0_mat, plot_path)
