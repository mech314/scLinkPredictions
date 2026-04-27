import pandas as pd
import glob
import os

rankings_base = "out/rankings"
data = []

for threshold_dir in glob.glob(f"{rankings_base}/*/"):
    threshold = os.path.basename(threshold_dir.rstrip("/"))
    
    for f in glob.glob(f"{threshold_dir}/*.tsv"):
        name = os.path.basename(f).replace(".tsv", "")
        parts = name.split("_")
        seed = parts[1]
        group = "_".join(parts[2:])
        df = pd.read_csv(f, sep="\t")
        mnr = 1 - df["rank"].median()
        data.append({
            "threshold": threshold,
            "seed": seed,
            "group": group,
            "mnr": mnr,
            "n": len(df)
        })

df_all = pd.DataFrame(data)

summary = df_all.groupby(["threshold", "group"])["mnr"].agg(["mean", "std"]).round(4)
summary.columns = ["mnr_mean", "mnr_std"]
summary = summary.reset_index()

print(summary.sort_values(["group", "mnr_mean"], ascending=[True, False]).to_string())
summary.to_csv("out/rankings/mnr_summary_all.csv", index=False)
print("\nSaved to out/rankings/mnr_summary_all.csv")