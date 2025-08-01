import numpy as np
import os
import sys
import json

from z3 import *
from .z3 import load_existing_models
from utils.z3_utils import model_to_abs, create_z3_args
from utils.new_api_utils import get_n_variations, get_lib_version, get_signature
from utils.misc import get_dir_in_root

def convert_model_to_abs(api, lib="torch"):
    api = get_lib_version(api, lib=lib)

    corpus_dir = "corpus_tf" if lib == "tf" else "corpus_torch"

    n_variations = get_n_variations(api, lib=lib)
    model_collection = {}
    if n_variations > 1:
        for i in range(1, n_variations + 1):
            model_collection[i] = {
                "models": []
            }
    else:
        model_collection[0] = {
            "models": []
        }

    for suffix in model_collection.keys():
        model_dir = os.path.join(get_dir_in_root(corpus_dir), f"{api}_{suffix}" if suffix > 0 else api)
        cur_sig = get_signature(api, lib=lib, suffix=suffix)
        model_collection[suffix]["z3_args"] = create_z3_args(cur_sig)
        if os.path.exists(model_dir):
            model_collection[suffix]["models"] = load_existing_models(model_dir, model_collection[suffix]["z3_args"])
        else:
            print(f"No existing models directory found for {api}_{suffix} (expected {model_dir}). Skipping.")
            continue

        if len(model_collection[suffix]['models']) == 0:
            print(f"No existing models found for {api}_{suffix} in {corpus_dir}. Skipping.")
            continue
        else:
            print(f"Loaded {len(model_collection[suffix]['models'])} existing models for {api} (suffix: {suffix})")

        for i, model in enumerate(model_collection[suffix]['models']):
            cur_sig = get_signature(api, lib=lib, suffix=suffix)
            abstract_input = model_to_abs(model, cur_sig, model_collection[suffix]['z3_args'])
            abstract_file = os.path.join(model_dir, f"abstract-{i}.json")
            with open(abstract_file, 'w') as f:
                json.dump(abstract_input, f, indent=2)


def main():    
    api = sys.argv[1]
    lib = sys.argv[2] if len(sys.argv) > 2 else "torch"

    # alias
    if lib == "tensorflow":
        lib = "tf"
    elif lib == "pytorch":
        lib = "torch"

    convert_model_to_abs(api, lib=lib)

if __name__ == "__main__":
    main()
