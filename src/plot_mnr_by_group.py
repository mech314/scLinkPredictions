import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("out/rankings/mnr_summary_all.csv")

# normalize threshold names
df["threshold"] = df["threshold"].str.replace("transe_", "").str.replace("scGPT_cosine_", "")
df["threshold"] = df["threshold"].str.replace("hpo_baseline", "baseline")

df["group"] = df["group"].replace({
    "NonUltraRare": "Rare",
    "RareDisease": "UltraRare"
})

groups = df["group"].unique()
n = len(groups)
ncols = 4
nrows = (n + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 4))
axes = axes.flatten()

order = ["baseline", "0.6", "0.7", "0.8", "0.9", "0.95", "0.99"]

for i, group in enumerate(sorted(groups)):
    ax = axes[i]
    sub = df[df["group"] == group].copy()
    sub["threshold"] = pd.Categorical(sub["threshold"], categories=order, ordered=True)
    sub = sub.sort_values("threshold")

    baseline_val = sub[sub["threshold"] == "baseline"]["mnr_mean"].values

    ax.errorbar(
        range(len(sub)),
        sub["mnr_mean"],
        yerr=sub["mnr_std"],
        fmt="o",
        color="#4c72b0",
        capsize=4,
        markersize=7
    )
    if len(baseline_val) > 0:
        ax.axhline(baseline_val[0], color="red", lw=1, ls="--", label="baseline")

    ax.set_title(group)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub["threshold"].tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("1-MNR")
    ax.legend(fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle("1-MNR by group — baseline vs augmented", fontsize=13)
plt.tight_layout()
plt.savefig("out/rankings/mnr_by_group.png", dpi=150, bbox_inches="tight")
print("Saved: out/rankings/mnr_by_group.png")