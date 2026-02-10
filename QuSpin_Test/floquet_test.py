import numpy as np
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.Floquet import Floquet
import matplotlib.pyplot as plt 

# Initial parameters
L = 8
J = 1.0
h = np.sqrt(2)

# Coupling lists
h_field = [[-h, i] for i in range(L)]
J_zz = [[-J, i, (i + 1) % L] for i in range(L)]  # PBC


# Floquet step durations (한 주기의 두 구간 길이)
switch_n = 50
tau_zz = 0.5; tau_x  = 0.5
dt_zz = tau_zz / switch_n
dt_x  = tau_x  / switch_n
T = tau_zz + tau_x

E_Ham_ = []
E_Floq_ = []
# Main Loop
for zblock in [-1, +1]:
    print(f"\n=== zblock = {zblock} ===")

    # basis in a given spin-inversion sector
    basis = spin_basis_1d(L=L, zblock=zblock)
    print("Hilbert space dim (sector) =", basis.Ns)

    # build the Hamiltonians (each static)
    # Overall / Two seperated step Hamiltonians
    H_overall = hamiltonian([["zz", J_zz], ["x", h_field]], [], basis=basis, dtype=np.float64)
    Hzz = hamiltonian([["zz", J_zz]], [], basis=basis, dtype=np.float64)
    Hx  = hamiltonian([["x",  h_field]], [], basis=basis, dtype=np.float64)

    # Hamiltonian Diagonalization
    E_Ham = H_overall.eigvalsh()
    Ewrap = np.sort(-np.angle(np.exp(-1j * E_Ham * T)) / T)
    print("First few Hamiltonian eigenvalues:", Ewrap[:6])
    E_Ham_.append(Ewrap)
    '''
    evo_dict_1 = {
        "H_list": [H_overall],  # step Hamiltonians (리스트로 전달)
        "dt_list": np.array([1.0], dtype=float)
    }
    Floq_1 = Floquet(evo_dict_1, UF=True, VF=True, thetaF=True)
    UF_1 = Floq_1.UF          # Floquet unitary U_F
    EF_1 = Floq_1.EF          # quasienergies
    thetaF_1 = Floq_1.thetaF  # eigenphases of U_F (보통 [-pi, pi] 범위)
    
    print("Drive period T =", Floq_1.T)
    print("First few Floquet quasienergies EF - Full H:", EF_1[:6])

    E_Ham_.append(EF_1)'''

    # Floquet from a LIST of step Hamiltonians (protocol type iii)
    # pass sparse matrices
    evo_dict = {
        "H_list": [2*Hzz, 2*Hx] * switch_n,  # step Hamiltonians (리스트로 전달)
        "dt_list": np.array([dt_zz, dt_x]*switch_n, dtype=float)
    }

    Floq = Floquet(evo_dict, UF=True, VF=True, thetaF=True)
    UF = Floq.UF          # Floquet unitary U_F
    EF = Floq.EF          # quasienergies (정렬된 값)
    thetaF = Floq.thetaF  # eigenphases of U_F (보통 [-pi, pi] 범위)

    print("Drive period T =", Floq.T)
    print("First few Floquet quasienergies EF:", EF[:6])

    E_Floq_.append(EF)

    # (참고) Floquet 고유값은 exp(-i * theta), 준에너지는 보통 EF ~ theta/T (mod 2pi/T)
    # QuSpin이 EF를 이미 준에너지 형태로 제공해 주지만,
    # 직접 확인하고 싶으면 대략 아래 관계를 떠올리면 됨.
    #   UF |phi> = exp(-i theta) |phi>
    #   quasienergy epsilon = theta / T  (mod 2pi/T)

plt.plot(
    np.arange(H_overall.Ns), E_Ham_[0] / L, marker="o", color="r", label="Hamiltonian"
)
plt.plot(
    np.arange(H_overall.Ns), E_Floq_[0] / L, marker="o", color="b", label="Floquet"
)
plt.xlabel("state number", fontsize=16)
plt.ylabel("energy", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.grid()
plt.tight_layout()
plt.savefig("example_test.png", bbox_inches="tight")


