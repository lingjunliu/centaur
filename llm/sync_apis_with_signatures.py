from utils.defaults import supported_paramtypes
from llm.tf_signatures import signatures as tf_signatures
from llm.torch_signatures import signatures as torch_signatures
from utils.misc import read_file_in_root
from utils.new_api_utils import get_api_suffix
import os
import json
import sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    lib = sys.argv[1] if len(sys.argv) > 1 else "torch"

    signatures = tf_signatures if lib == "tf" else torch_signatures
    original_apis = read_file_in_root(f"{lib}_apis.txt")
    variations = set(signatures.keys())

    signatures_file = os.path.join(CUR_DIR, "../signatures.json")
    with open(signatures_file, "r") as f:
        original_signatures = json.load(f)

    finalized_apis = set()
    finalized_variations = set()
    for variation in variations:
        api, suffix = get_api_suffix(variation)
        if api not in original_apis:
            continue
        sig = signatures[variation]
        supported = True
        
        if api.startswith("tf.raw_ops"):    # tf.raw_ops should have no args, only kwargs
            for arg, domain in sig["args"].items():
                sig["kwargs"][arg] = domain
            sig["args"] = {}

        for key in ["args", "kwargs"]:
            for arg, domain in sig[key].items():
                if domain not in supported_paramtypes:
                    supported = False
                    print(f"API {api} has unsupported argument {arg} with domain {domain}")
                    break
        
        if supported:
            finalized_apis.add(api)
            finalized_variations.add(variation)
    
    with open(os.path.join(CUR_DIR, f"{lib}_finalized_apis.txt"), "w") as f:
        f.write("\n".join(sorted(finalized_apis)))

    for variation in sorted(finalized_variations):
        original_signatures[variation] = signatures[variation]
    
    with open(signatures_file, "w") as f:
        json.dump(original_signatures, f, indent=4)

    variations_file = os.path.join(CUR_DIR, f"../{lib}_variations.txt")
    with open(variations_file, "w") as f:
        f.write("\n".join(sorted(finalized_variations)))

if __name__ == "__main__":
    main()