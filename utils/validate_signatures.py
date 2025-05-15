import pandas as pd
import json, sys

def dict_equal(d1, d2):
    if len(d1) != len(d2):
        print(f"Length mismatch: {len(d1)} vs {len(d2)}")
        return False
    for k in d1:
        if k not in d2 or d1[k] != d2[k]:
            print(f"Key mismatch: {k} -> {d1[k]} vs {d2.get(k)}")
            return False
    return True

def main():
    if len(sys.argv) < 3:
        print("Usage: python validate_signatures.py <path_to_reference_signatures.csv> <path_to_existing_signatures.json>")
        return
    
    df = pd.read_csv(sys.argv[1])
    entries = list(df["Signature"])
    apis = list(df["API"])

    with open(sys.argv[2], 'r') as f:
        existing = json.load(f)

    cnt = 0
    for api, entry in zip(apis, entries):
        acc_str = entry.replace("'", "\"")
        new_dict = json.loads(acc_str)
        if api in existing and not dict_equal(existing[api], new_dict):
            print(f"API {api} has different entry in signatures.json.")
            print(f"Existing: {existing[api]}")
            print(f"Reference: {new_dict}")
            print()
            cnt += 1

    print(f"Total APIs with different entries: {cnt}")

if __name__ == "__main__":
    main()