import numpy as np
import matplotlib.pyplot as plt

g = np.load("gate_sample.npy")   # shape [B, C, P]
g0 = g[0]                        # first sample -> [C, P]

plt.figure()
plt.imshow(g0, aspect="auto")
plt.colorbar(label="Gate value g (0..1)")
plt.xlabel("Patch index")
plt.ylabel("Channel index (variable)")
plt.title("FuzzGate Heatmap (one test sample): Channels × Patches")
plt.tight_layout()
plt.savefig("gate_heatmap.png", dpi=200)
plt.show()
print("Saved gate_heatmap.png")