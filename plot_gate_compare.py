import numpy as np
import matplotlib.pyplot as plt

g0 = np.load("gate_mask0.npy")[0]     # [C, P]
g30 = np.load("gate_mask30.npy")[0]   # [C, P]

vmin = min(g0.min(), g30.min())
vmax = max(g0.max(), g30.max())

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.imshow(g0, aspect="auto", vmin=vmin, vmax=vmax)
plt.title("Gate Heatmap (mask_rate=0.0)")
plt.xlabel("Patch index")
plt.ylabel("Channel index")

plt.subplot(1, 2, 2)
plt.imshow(g30, aspect="auto", vmin=vmin, vmax=vmax)
plt.title("Gate Heatmap (mask_rate=0.3)")
plt.xlabel("Patch index")
plt.ylabel("Channel index")

plt.tight_layout()
plt.savefig("gate_heatmap_compare.png", dpi=200)
print("Saved gate_heatmap_compare.png")