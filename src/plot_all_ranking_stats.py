import pandas as pd
import glob
import json
import os
import matplotlib.pyplot as plt

rankings_base = "out/rankings"
models_base = "out/PyKeenOut"

# 1-MNR
mnr_data = []
for threshold_dir in glob.glob(f"{rankings_base}/*/"):
    threshold = os.path.basename(threshold_dir.rstrip("/"))
    for f in glob.glob(f"{threshold_dir}/*.tsv"):
        name = os.path.basename(f).replace(".tsv", "")
        parts = name.split("_")
        seed = parts[1]
        group = "_".join(parts[2:])
        df = pd.read_csv(f, sep="\t")
        mnr = 1 - df["rank"].median()
        mnr_data.append({"threshold": threshold, "seed": seed, "group": group, "mnr": mnr})

df_mnr = pd.DataFrame(mnr_data)
mnr_summary = df_mnr.groupby(["threshold", "group"])["mnr"].mean().reset_index()

# Hits
hits_data = []
for results_json in glob.glob(f"{models_base}/*/models/*/*/results.json"):
    parts = results_json.split("/")
    threshold = parts[2]
    seed = parts[4]
    with open(results_json) as f:
        data = json.load(f)
    b = data["metrics"]["both"]["realistic"]
    hits_data.append({
        "threshold": threshold,
        "seed": seed,
        "hits@1": b["hits_at_1"],
        "hits@10": b["hits_at_10"],
        "mrr": b["inverse_harmonic_mean_rank"]
    })

df_hits = pd.DataFrame(hits_data)
hits_summary = df_hits.groupby("threshold")[["hits@1", "hits@10", "mrr"]].mean().reset_index()

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ["hits@1", "hits@10", "mrr"]
titles = ["Hits@1", "Hits@10", "MRR"]

for ax, metric, title in zip(axes, metrics, titles):
    baseline_val = hits_summary[hits_summary["threshold"] == "transe_baseline"][metric].values
    for _, row in hits_summary.iterrows():
        color = "gray" if row["threshold"] == "transe_baseline" else "#4c72b0"
        ax.scatter(row["threshold"].replace("transe_", ""), row[metric], color=color, s=80)
    if len(baseline_val) > 0:
        ax.axhline(baseline_val[0], color="red", lw=1, ls="--", label="baseline")
    ax.set_title(title)
    ax.set_xlabel("threshold")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("TransE — baseline vs augmented", fontsize=12)
plt.tight_layout()
plt.savefig("out/rankings/hits_scatter.png", dpi=150, bbox_inches="tight")
print("Saved: out/rankings/hits_scatter.png")