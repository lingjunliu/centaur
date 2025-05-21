from utils.misc import read_pkl, get_tmp_dir
from utils.api_utils import get_driver, get_signatures
from generator.input_generators import concretize_input, abstract_print
from eval.oracle import oracle_diff
import sys, os
import numpy as np
from rapidfuzz import fuzz

def main():
    A_TOL = 1e-02
    PRINT_INDICES = True
    THRESHOLD = 80  # Thrshold for similarity for exception messages
    
    if len(sys.argv) < 4:
        print("Usage: python analyze_inputs.py <api> <lib> <low> <index> <detailed, default: False>")
        return
    
    api = sys.argv[1]
    lib = sys.argv[2] if len(sys.argv) > 2 else "torch"
    low = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    ind = int(sys.argv[4]) if len(sys.argv) > 4 else None
    detailed = True if len(sys.argv) > 5 and sys.argv[5] == "detailed" else False # for detailed output with all indices with max_diff
    
    # alias
    if lib == "pytorch":
        lib = "torch"
    elif lib == "tensorflow":
        lib = "tf"
    
    # Directory containing the input files
    tmp = get_tmp_dir()
    input_file = os.path.join(tmp, "fuzz_inputs", f"{api}_inputs.pkl")
    oracle_folder = f"oracle_results_{lib}"
    oracle_file = os.path.join(tmp, oracle_folder, f"{api}.pkl")
    
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist.")
        return
    if not os.path.exists(oracle_file):
        print(f"Oracle file {oracle_file} does not exist.")
        return
    generated_inputs = read_pkl(input_file)
    oracle_results = read_pkl(oracle_file)
    
    if ind is not None:
        if ind < 0 or ind >= len(generated_inputs):
            print(f"Index {ind} out of range for generated inputs.")
            return
        best_distance, abs_input, seed = generated_inputs[ind]
        
        if ind < len(oracle_results) + low:
            oracle_result = oracle_results[ind-low]
            
            if oracle_result[0] == "nominal":
                print(f"Input {ind} is nominal.")
            elif oracle_result[0] == "invalid":
                print(f"Input {ind} is invalid: {oracle_result[1]}")
            else:
                print(f"{ind}: {'\n'.join([str(x) for x in oracle_result])}")
            
        print("\nRe running the oracle...")
        driver = get_driver(api, lib=lib)
        signature = get_signatures()[api]
        rng = np.random.default_rng(seed)
        input_dict = concretize_input(abs_input, signature, rng)
        diff_oracle_result = oracle_diff(driver, input_dict, atol=A_TOL, detailed=detailed)
        
        print(f"\nOracle result: {diff_oracle_result}")
        print(f"\nAbstract input (seed {seed}): {abstract_print(abs_input, signature)}")
    else:
        err_count = {}
        groups = {}
        inconsistencies = {}
        print(f"Total inputs: {len(generated_inputs)}")
        print(f"Total oracles: {len(oracle_results)}")
        for i, oracle_result in enumerate(oracle_results):
            if oracle_result[0] == "nominal":
                continue
            elif oracle_result[0] == "invalid":
                continue
            elif oracle_result[0] == "inconsistent":
                if oracle_result[1] not in inconsistencies:
                    inconsistencies[oracle_result[1]] = []
                inconsistencies[oracle_result[1]].append(low+i)
            else:
                if oracle_result[0] not in err_count:
                    err_count[oracle_result[0]] = {}
                    groups[oracle_result[0]] = {}
                
                found_group = False
                for group_rep in groups[oracle_result[0]].keys():
                    if fuzz.ratio(oracle_result[1], group_rep) > THRESHOLD:
                        groups[oracle_result[0]][group_rep].append(oracle_result[1])
                        err_count[oracle_result[0]][group_rep].append(low+i)
                        found_group = True
                        break
                
                if not found_group:
                    err_count[oracle_result[0]][oracle_result[1]] = [low+i]
                    groups[oracle_result[0]][oracle_result[1]] = [oracle_result[1]]

        for err_type, err_dict in err_count.items():
            print(f"\nError type: {err_type}")
            for err_msg, indices in err_dict.items():
                print(f"  Error message: {err_msg}")
                print(f"  Indices: {indices}" if PRINT_INDICES else f"  Count: {len(indices)}")
        
        if len(inconsistencies) > 0 :
            print("\nInconsistencies:")
        for inconsistency, indices in inconsistencies.items():
            print(f"  Inconsistency: {inconsistency} | Indices: {indices}" if PRINT_INDICES else f"  Count: {len(indices)}\n")            
if __name__ == "__main__":
    main()