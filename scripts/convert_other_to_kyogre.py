from pathlib import Path
import pandas as pd

LABELS_CSV = Path("data/pokemon_photos/labels.csv")

def main():
    df = pd.read_csv(LABELS_CSV)

    # show current counts
    print("Before:")
    print(df["label"].value_counts())

    # convert
    mask = df["label"] == "other_card"
    df.loc[mask, "label"] = "kyogre_v_037_159"

    print("\nAfter:")
    print(df["label"].value_counts())

    df.to_csv(LABELS_CSV, index=False)
    print(f"\n[INFO] Saved updated labels to {LABELS_CSV}")

if __name__ == "__main__":
    main()