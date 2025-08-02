import pandas as pd
from pathlib import Path
from typing import List
import re

# List of experiment directories
env_paths: List[Path] = [Path('exp/antsoccer-exp'), Path('exp/cube-exp')]

# File pattern and regex
glob_pattern = "seed_*_alpha_*_*_*"
regex = re.compile(r"seed_(.*?)_alpha_(.*?)_")

# Collect all rows from all eval.csv files
rows = []

for subdir in env_paths[0].glob(glob_pattern):
    match = regex.match(subdir.name)
    if match:
        seed = match.group(1)
        alpha = match.group(2)

        eval_path = subdir / "eval.csv"
        if eval_path.exists():
            try:
                eval_df = pd.read_csv(eval_path)

                for _, row in eval_df.iterrows():
                    rows.append({
                        "seed": seed,
                        "alpha": alpha,
                        "step": row["step"],
                        "success": row["success"],
                    })
            except Exception as e:
                print(f"⚠️ Error reading {eval_path}: {e}")
        else:
            print(f"⚠️ Missing: {eval_path}")

# Combine all into a DataFrame
df = pd.DataFrame(rows)
df.to_csv('csvs/antsoccer.csv')
print(df)
