import os, json
from llm.signatures import signatures

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    sig_file = os.path.join(CUR_DIR, f"../signatures.json")
    if os.path.exists(sig_file):
        with open(sig_file, "r") as f:
            existing_signatures = json.load(f)

        for api, sig in signatures.items():
            if api not in existing_signatures:
                if api.startswith("torch."):
                    continue
                existing_signatures[api] = sig
            elif len(sig.keys()) > len(existing_signatures[api].keys()):
                existing_signatures[api] = sig
    else:
        existing_signatures = signatures
    
    with open(sig_file, "w") as f:
        json.dump(existing_signatures, f)

if __name__ == "__main__":
    main()