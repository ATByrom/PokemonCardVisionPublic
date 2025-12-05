import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CARDS_CSV = BASE_DIR / "data" / "pokemon_tcg_hf" / "cards.csv"

def main():
    df = pd.read_csv(CARDS_CSV)
    print(f"Total rows in CSV: {len(df)}")

    missing = []

    for i, row in df.iterrows():
        lp = str(row.get("local_path", "")).strip()

        # skip rows that genuinely have no path (shouldn't happen for your 10k)
        if not lp:
            missing.append((i, row.get("id", ""), row.get("name", ""), "<EMPTY>"))
            continue

        img_path = BASE_DIR / lp.replace("\\", "/")

        if not img_path.exists():
            missing.append((i, row.get("id", ""), row.get("name", ""), lp))

    if not missing:
        print("✅ All local_path files exist on disk.")
    else:
        print(f"❌ Found {len(missing)} missing files:")
        for idx, cid, name, lp in missing:
            print(f"  Row {idx}: id={cid}, name={name}, local_path='{lp}'")

if __name__ == "__main__":
    main()