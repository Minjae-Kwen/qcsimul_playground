import matplotlib.pyplot as plt
import numpy as np

g_per_J_list = [0.2 * i for i in range(2, 9)]
E_mat = [0.9340, 0.9363, 0.9195, 0.8941, 0.8901, 0.9088, 0.9077]

plt.plot(g_per_J_list, E_mat, marker="o", color="green")
plt.xlabel("g/J", fontsize=16)
plt.ylabel(r"$E/E_0$", fontsize=16)
plt.ylim((0.68, 1.02))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
#plt.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
plt.savefig("TFIM_E_gJ.png", bbox_inches="tight")