from utils.defaults import supported_paramtypes
from llm.tf_signatures import signatures as tf_signatures
from llm.torch_signatures import signatures as torch_signatures
from utils.misc import read_file_in_root
import os
import json
import sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    lib = sys.argv[1] if len(sys.argv) > 1 else "torch"

    signatures = tf_signatures if lib == "tf" else torch_signatures
    apis = read_file_in_root(f"{lib}_apis.txt")

    if os.path.exists(os.path.join(CUR_DIR, f"failed_sig_{lib}.txt")):
        with open(os.path.join(CUR_DIR, f"failed_sig_{lib}.txt"), "r") as f:
            failed_apis = [line.strip() for line in f.readlines()]
    else:
        failed_apis = []

    signatures_file = os.path.join(CUR_DIR, "../signatures.json")
    with open(signatures_file, "r") as f:
        original_signatures = json.load(f)

    finalized_apis = []
    for api in apis:
        if api in failed_apis:
            continue
        if api in signatures:
            sig = signatures[api]
            supported = True
            
            for key in ["args", "kwargs"]:
                for arg, domain in sig[key].items():
                    if domain not in supported_paramtypes:
                        supported = False
                        print(f"API {api} has unsupported argument {arg} with domain {domain}")
                        break
            
            if supported:
                finalized_apis.append(api)
        else:
            print(f"API {api} not found in signatures")
    
    with open(os.path.join(CUR_DIR, f"{lib}_finalized_apis.txt"), "w") as f:
        f.write("\n".join(finalized_apis))

    for api in finalized_apis:
        original_signatures[api] = signatures[api]
    
    with open(signatures_file, "w") as f:
        json.dump(original_signatures, f, indent=4)

if __name__ == "__main__":
    main()