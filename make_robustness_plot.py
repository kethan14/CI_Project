import pandas as pd
import matplotlib.pyplot as plt

# Fill in the numbers you already measured
data = [
    {"mask_rate": 0.0, "model": "PatchTST",  "mse": 0.2878046929836273, "mae": 0.34036460518836975, "rse": 0.5104787945747375},
    {"mask_rate": 0.1, "model": "PatchTST",  "mse": 0.30402296781539917, "mae": 0.3603231608867645,  "rse": 0.5246648192405701},
    {"mask_rate": 0.3, "model": "PatchTST",  "mse": 0.3698963224887848,  "mae": 0.42332935333251953, "rse": 0.5787203311920166},

    {"mask_rate": 0.0, "model": "FuzzGate-TST (V1)", "mse": 0.2878046929836273, "mae": 0.34036460518836975, "rse": 0.5104787945747375},
    {"mask_rate": 0.1, "model": "FuzzGate-TST (V1)", "mse": 0.30402296781539917, "mae": 0.3603231608867645,  "rse": 0.5246648192405701},
    {"mask_rate": 0.3, "model": "FuzzGate-TST (V1)", "mse": 0.3698963224887848,  "mae": 0.42332935333251953, "rse": 0.5787203311920166},
]

df = pd.DataFrame(data).sort_values(["mask_rate", "model"])
df.to_csv("robustness_results.csv", index=False)
print("Saved: robustness_results.csv")

# Plot MAE vs mask_rate
plt.figure()
for model_name in df["model"].unique():
    sub = df[df["model"] == model_name].sort_values("mask_rate")
    plt.plot(sub["mask_rate"], sub["mae"], marker="o", label=model_name)

plt.xlabel("Mask rate (missingness)")
plt.ylabel("MAE")
plt.title("Robustness to Missingness: MAE vs Mask Rate")
plt.legend()
plt.tight_layout()
plt.savefig("robustness_mae_plot.png", dpi=200)
print("Saved: robustness_mae_plot.png")
plt.show()