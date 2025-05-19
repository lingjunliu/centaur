from llm.signatures import signatures
from llm.generate_valid_inputs import get_torch_api
import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    with open(f"{CUR_DIR}/api_full.txt") as f:
        all_apis = set([api.strip() for api in f.readlines()])
    sig_apis = set()
    for torch_api, signature in signatures.items():
        sig_apis.add(torch_api)

    sig_apis = sig_apis - all_apis
    
    existing_apis = []
    with open(os.path.join(CUR_DIR, "../reference_signatures.csv"), "r") as f:
        for line in f.readlines():
            tokens = line.strip().split(",")
            if tokens[0] == "API":
                continue
            existing_apis.append(get_torch_api(tokens[0]))
            
    existing_sig = set(existing_apis)
    
    no_sig = all_apis - sig_apis - existing_sig
    
    with open(f"{CUR_DIR}/needs_sig.txt", "w") as f:
        f.write("\n".join(list(no_sig)))
    
    print(f"\nGenerated {len(sig_apis)}/{len(all_apis)} signatures, previously had {len(existing_sig)} signatures.\n")
    print(f"Total: {len(sig_apis)+len(existing_sig)}/{len(all_apis)} signatures")
    
if __name__ == "__main__":
    main()