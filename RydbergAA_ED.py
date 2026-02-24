import os
import numpy as np
import scipy.linalg as la
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.Floquet import Floquet
import matplotlib.pyplot as plt
from tqdm import tqdm

# Method Definition

def hamiltonian_rydberg1(basis, Nsys):
    #Ntot = Nsys + Nanc
    #dimS = 2**Nsys; dimA = 2**Nanc

    # 1. Prep for system Hamiltonian, etc
    static_coupling = [
        ["x", [[0.5, i] for i in range(Nsys)]]
    ]
    H_coupling = hamiltonian(static_coupling, [], basis=basis, dtype=np.float64, check_herm=False, check_symm=False)

    static_detuning = [
        ["I", [[-0.5, i] for i in range(Nsys)]], 
        ["z", [[-0.5, i] for i in range(Nsys)]]
    ]
    H_detuning = hamiltonian(static_detuning, [], basis=basis, dtype=np.float64, check_herm=False, check_symm=False)

    blockade_pairs = [(i, j) for i in range(Nsys-1) for j in range(i+1, Nsys)]
    static_blockade = [
        ["II", [[0.25/(j-i)**6, i, j] for (i,j) in blockade_pairs]],
        ["zI", [[0.25/(j-i)**6, i, j] for (i,j) in blockade_pairs]],
        ["Iz", [[0.25/(j-i)**6, i, j] for (i,j) in blockade_pairs]],
        ["zz", [[0.25/(j-i)**6, i, j] for (i,j) in blockade_pairs]],
    ]
    H_blockade = hamiltonian(static_blockade, [], basis=basis, dtype=np.float64, check_herm=False, check_symm=False)

    return H_coupling, H_detuning, H_blockade

def n_site(psi, basis, site):
    probs = np.abs(psi)**2
    states = basis.states.astype(np.uint64)  
    occ = ((states >> site) & 1).astype(np.float64)
    return float(np.dot(probs, occ))

    #sz = hamiltonian([["z", [[1.0, site]]]], [], basis=basis, dtype=np.float64)
    #return 0.5*(1.0 + np.vdot(psi, sz.dot(psi)).real)

def n_twosite(psi, basis, site1, site2):
    '''
    probs = np.abs(psi)**2
    states = basis.states.astype(np.uint64)

    occ1 = (states >> site1) & 1
    occ2 = (states >> site2) & 1
    pair = (occ1 & occ2).astype(np.float64)  # 1 only if both are 1

    return float(np.dot(probs, pair))
    '''
    s_z_sq = hamiltonian([["zz", [[1.0, site1, site2]]]], [], basis=basis, dtype=np.float64, check_herm=False, check_symm=False)
    s_z1 = hamiltonian([["z", [[1.0, site1]]]], [], basis=basis, dtype=np.float64, check_herm=False, check_symm=False)
    s_z2 = hamiltonian([["z", [[1.0, site2]]]], [], basis=basis, dtype=np.float64, check_herm=False, check_symm=False)
    return 0.25*(1.0 + np.vdot(psi, s_z_sq.dot(psi)).real + np.vdot(psi, s_z1.dot(psi)).real + np.vdot(psi, s_z2.dot(psi)).real)
    

# Initial parameters
Nsys, Nanc = 13, 0
Ntot = Nsys + Nanc
#dimS = 2**Nsys; dimA = 2**Nanc

Rb_over_a = 1.15 # Unit: NoDim
Omega = 2.0 # Unit: rad/us
Delta_range = np.arange(-6.0, 12.1, 2.0) # Unit: rad/us

V0_over_a6 = Omega * Rb_over_a**6 # Unit: rad/us

basis = spin_basis_1d(L=Ntot)
H_coupling, H_detuning, H_blockade = hamiltonian_rydberg1(basis, Nsys)

blockade_array = np.zeros((len(Delta_range), Nsys), dtype=np.float64)
for idx, Delta in tqdm(enumerate(Delta_range), total=len(Delta_range)):
    Hsys = Omega * H_coupling + Delta * H_detuning + V0_over_a6 * H_blockade
    E, V = Hsys.eigsh(k=1, which="SA", return_eigenvectors=True)
    E0 = E[0]; psi0 = V[:, 0]
    #breakpoint()

    blockade_array[idx, 0] = 1.0
    for k in range(1, Nsys):
        N_k = Nsys - k
        for site1 in range(N_k):
            site2 = site1 + k
            blockade_array[idx, k] += ( n_twosite(psi0, basis, site1, site2) ) / N_k

#breakpoint()

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


def plot_colormap(Delta_list, Nsys, array4color, Omega, Rb_over_a, plot_path):
    #sitedist_list = np.arange(1, Nsys)
    dDelta = Delta_list[1] - Delta_list[0]

    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=150)
    im = ax.imshow(
        array4color,
        origin="lower",
        extent=[-0.5, Nsys - 0.5, Delta_list[0]-dDelta/2, Delta_list[-1]+dDelta/2],
        aspect="auto",
        interpolation="none", 
        cmap = "gray_r",
        vmin=0.0, vmax=1.0
    )

    ax.set_xlabel("Site Distance, d")
    ax.set_ylabel(r"$\Delta$ (rad/$\mu$s)")

    ax.set_xticks(np.arange(0, Nsys, 2))
    ax.set_yticks(Delta_list)
    ax.set_xticks(np.arange(-0.5, Nsys, 1), minor=True)
    ax.set_yticks(Delta_list - dDelta/2, minor=True) 
    ax.grid(which="minor", color="k", linewidth=0.4, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"Avg($\langle n_i n_{i+d}\rangle$)")

    info = rf"$\Omega = {Omega:.3g}$ (rad/$\mu$s)" + "\n" + rf"$R_b/a = {Rb_over_a:.3g}$"
    ax.text(
        0.02, 0.02, info,
        transform=ax.transAxes,   # axes fraction 좌표
        ha="left", va="bottom",
        fontsize=9,
        color="black",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.75)
    )

    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")

plot_colormap(Delta_range, Nsys, blockade_array, Omega, Rb_over_a, f"blockade_colormap_Rbovera{Rb_over_a:.2f}.png")