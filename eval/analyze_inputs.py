from utils.misc import read_pkl, get_tmp_dir
from utils.api_utils import get_driver, get_signatures
from generator.input_generators import concretize_input, abstract_print
from eval.oracle import oracle_diff
import sys, os
import numpy as np

def main():
    A_TOL = 1e-02
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_inputs.py <api>")
        return
    
    api = sys.argv[1]
    ind = int(sys.argv[2]) if len(sys.argv) > 2 else None
    # Directory containing the input files
    tmp = get_tmp_dir()
    input_file = os.path.join(tmp, "fuzz_inputs", f"{api}_inputs.pkl")
    oracle_file = os.path.join(tmp, "oracle_results", f"{api}.pkl")
    
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
        
        if ind < len(oracle_results):
            oracle_result = oracle_results[ind]
            
            if oracle_result[0] == "nominal":
                print(f"Input {ind} is nominal.")
            elif oracle_result[0] == "invalid":
                print(f"Input {ind} is invalid: {oracle_result[1]}")
            else:
                print(f"{ind}: {'\n'.join([str(x) for x in oracle_result])}")
            
        print("\nRe running the oracle...")
        driver = get_driver(api)
        signature = get_signatures()[api]
        rng = np.random.default_rng(seed)
        input_dict = concretize_input(abs_input, signature, rng)
        diff_oracle_result = oracle_diff(driver, signature, input_dict, atol=A_TOL)
        
        print(f"\nOracle result: {diff_oracle_result}")
        print(f"\nAbstract input (seed {seed}): {abstract_print(abs_input, signature)}")
    else:
        print(f"Total inputs: {len(generated_inputs)}")
        print(f"Total oracles: {len(oracle_results)}")
        for i, oracle_result in enumerate(oracle_results):
            if oracle_result[0] == "nominal":
                continue
            elif oracle_result[0] == "invalid":
                continue
            else:
                print(f"{i}: {'\n'.join([str(x) for x in oracle_result])}\n")

if __name__ == "__main__":
    main()