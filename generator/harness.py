import time
import os
import pickle
import sys
import numpy as np

from .ea import Configuration, Mutator, optimize
from .definitions import get_definition
from .input_generators import get_random_input
from learner.invariant_inference import infer_invariants
from utils.api_utils import get_driver
from utils.misc import create_subdir, get_tmp_dir
from utils.defaults import domain_limits_torch as domain_limits
from eval.oracle import oracle_crash

############### api definition examples ################

# Scatter

scatter_definition = {
    "signature":    {
                        "input": "tensor",
                        "dim": "integer",
                        "index": "tensor",
                        "src": "tensor"
                    },
    # Set manually
    # "ruleset":  set([
    #                     ('rule_2', 'input', 'dim'),
    #                     ('rule_2', 'index', 'dim'),
    #                     ('rule_2', 'src', 'dim'),
    #                     ('rule_3', 'input', 'src'),
    #                     ('rule_3', 'input', 'index'),
    #                     ('rule_3', 'src', 'index'),
    #                     ('rule_4', 'input', 'src'),
    #                     ('rule_5', 'input', 'index')
    #                 ]),
    # Using invariant inference
    "ruleset":  infer_invariants("scatter")[0],
    # Easy
    # "random_candidate": {
    #                         "input": np.random.rand(2,4).astype(np.float32),
    #                         "dim": 7,
    #                         "index": np.random.rand(2,1).astype(np.int64),
    #                         "src": np.random.rand(2,1,3).astype(np.float32)
    #                     },
    # Hard
    "random_candidate": {
                            "input": np.random.rand(2,4, 343, 10).astype(np.float32),
                            "dim": 7,
                            "index": np.random.rand(2).astype(np.int64),
                            "src": np.random.rand(2,1).astype(np.complex64)
                        },
    "arg_order": ['input', 'dim', 'index', 'src'],
    # set limits based on index of argument in the order
    # three lists per argument: value list, dtype list, range list
    # min_num, max_num, min_size, max_size
    "limits":   [            
                    domain_limits['tensor'], # input value
                    domain_limits['tensor_dtype'],  # input dtype
                    domain_limits['tensor_value_range'],    # input range of values
                    domain_limits['integer'], # dim value
                    domain_limits['integer_dtype'], # dim dtype, only allow integer types
                    domain_limits['integer_value_range'],    # dim, range of values not really needed, added for symmetry
                    domain_limits['tensor'], # index value
                    domain_limits['tensor_dtype'],  # index dtype
                    domain_limits['tensor_value_range'],    # index range of values
                    domain_limits['tensor'], # src value
                    domain_limits['tensor_dtype'],  # src dtype
                    domain_limits['tensor_value_range'],    # src range of values
                ]
}

# atan2

atan2_definition = {
    "signature":    {
                        "input": "tensor",
                        "other": "tensor"
                    },
    "ruleset":  set([
                        ('rule_3', 'input', 'other'),
                        ('rule_6', 'input', 'other'),
                        ('rule_1', 'input', 'other'),
                        ('rule_4', 'input', 'other'),
                        ('rule_6', 'other', 'input')
                    ]),
    "random_candidate": {
                            "input": np.random.rand(2,4,343,10,1).astype(np.float32),
                            "other": np.random.rand(2,1,3).astype(np.int64)
                        },
    "arg_order": ['input', 'other'],
    # set limits based on index of argument in the order
    # three lists per argument: value list, dtype list, range list
    # min_num, max_num, min_size, max_size
    "limits":   [            
                    domain_limits['tensor'], # input value
                    domain_limits['tensor_dtype'],  # input dtype
                    domain_limits['tensor_value_range'],    # input range of values
                    domain_limits['tensor'], # other value
                    domain_limits['tensor_dtype'],  # other dtype
                    domain_limits['tensor_value_range'],    # other range of values
                ]
}

# matmul

matmul_definition = {
    "signature":    {
                        "input": "tensor",
                        "other": "tensor"
                    },
    "ruleset":  set([
                        ('rule_6', 'input', 'other'),
                        ('rule_4', 'input', 'other')                        
                    ]),
    "random_candidate": {
                            "input": np.random.rand(2,4,343,10,1).astype(np.float32),
                            "other": np.random.rand(2,1,3).astype(np.int64)
                        },
    "arg_order": ['input', 'other'],
    # set limits based on index of argument in the order
    # three lists per argument: value list, dtype list, range list
    # min_num, max_num, min_size, max_size
    "limits":   [            
                    domain_limits['tensor'], # input value
                    domain_limits['tensor_dtype'],  # input dtype
                    domain_limits['tensor_value_range'],    # input range of values
                    domain_limits['tensor'], # other value
                    domain_limits['tensor_dtype'],  # other dtype
                    domain_limits['tensor_value_range'],    # other range of values
                ]
}

# argmin

argmin_definition = {
    "signature":    {
                        "input": "tensor",
                        "dim": "integer",
                    },
    "ruleset":  set([
                        ('rule_2', 'input', 'dim')                        
                    ]),
    "random_candidate": {
                            "input": np.random.rand(2,4,343,10,1).astype(np.float32),
                            "dim": 10,
                            "keepdim": True
                        },
    "arg_order": ['input', 'dim'],
    # set limits based on index of argument in the order
    # three lists per argument: value list, dtype list, range list
    # min_num, max_num, min_size, max_size
    "limits":   [            
                    domain_limits['tensor'], # input value
                    domain_limits['tensor_dtype'],  # input dtype
                    domain_limits['tensor_value_range'],    # input range of values
                    domain_limits['integer'], # dim value
                    domain_limits['integer_dtype'],  # dim dtype
                    domain_limits['integer_value_range'],    # dim range of values
                ]
}

# conv_transpose2d

conv_transpose2d_definition = {
    "signature":    {
                        "input": "tensor",
                        "weight": "tensor",
                        "stride": "integer",
                        "padding": "integer"
                    },
    # Set manually
    # "ruleset":  set([
    #                     ('rule_3', 'input', 'weight'),
    #                     ('rule_4', 'input', 'weight'),
    #                     ('rule_7', 'weight', 'input')
    #                 ]),
    # Using invariant inference
    "ruleset":  infer_invariants("conv_transpose2d")[0],
    "random_candidate": {
                            "input": np.random.rand(2,4,343,10,1).astype(np.float32),
                            "weight": np.random.rand(2,343,1).astype(np.float32),
                            "stride": 10,
                            "padding": 1
                        },
    "arg_order": ['input', 'weight', 'stride', 'padding'],
    # set limits based on index of argument in the order
    # three lists per argument: value list, dtype list, range list
    # min_num, max_num, min_size, max_size
    "limits":   [            
                    domain_limits['tensor'], # input value
                    domain_limits['tensor_dtype'],  # input dtype
                    domain_limits['tensor_value_range'],    # input range of values
                    domain_limits['tensor'], # weight value
                    domain_limits['tensor_dtype'],  # weight dtype
                    domain_limits['tensor_value_range'],    # weight range of values
                    domain_limits['integer'], # stride value
                    domain_limits['integer_dtype'],  # stride dtype
                    domain_limits['integer_value_range'],    # stride range of values
                    domain_limits['integer'], # padding value
                    domain_limits['integer_dtype'],  # padding dtype
                    domain_limits['integer_value_range'],    # padding range of values
                ]
}


############### example definitions ################
# For testing single rules

# define rule 1
r1_definition = {
    "signature":    {
                        "input": "tensor",
                        "other": "tensor"
                    },
    "ruleset":  set([
                    (2, 'rule_1', 'input', 'other')
                ]),
    "random_candidate": {
                            "input": np.random.rand(3,2,5,9,343,2),
                            "other": np.random.rand(4, 2)
                        },
    "arg_order": ['input', 'other'],
    # set limits based on index of argument in the order
    # three lists per argument: value list, dtype list, range list
    # min_num, max_num, min_size, max_size
    "limits":   [            
                    domain_limits['tensor'], # input value
                    domain_limits['tensor_dtype'],  # input dtype
                    domain_limits['tensor_value_range'], # input value range
                    domain_limits['tensor'], # other value
                    domain_limits['tensor_dtype'],  # other dtype
                    domain_limits['tensor_value_range'] # other value range
                ]
}

# define rule 2
r2_definition = {
    "signature":    {
                        "input": "tensor",
                        "dim": "integer"
                    },
    "ruleset":  set([
                    (2, 'rule_2', 'input', 'dim')
                ]),
    "random_candidate": {
                            "input": np.random.rand(3,2,5,9,343,2),
                            "dim": 25
                        },
    "arg_order": ['input', 'dim'],
    # set limits based on index of argument in the order
    # three lists per argument: value list, dtype list, range list
    # min_num, max_num, min_size, max_size
    "limits":   [            
                    domain_limits['tensor'], # input value
                    domain_limits['tensor_dtype'],  # input dtype
                    domain_limits['tensor_value_range'], # input value range
                    domain_limits['integer'], # dim value
                    domain_limits['integer_dtype'],  # dim dtype, integer only
                    domain_limits['integer_value_range']    # dim value range, we don't use it for integers, it is here just for cohesion
                ]
}

# define rule 3
r3_definition = {
    "signature":    {
                        "input": "tensor",
                        "other": "tensor"
                    },
    "ruleset":  set([
                    (2, 'rule_3', 'input', 'other')
                ]),
    "random_candidate": {
                            "input": np.random.rand(3,2,5,9,343,2),
                            "other": np.random.rand(4, 2)
                        },
    "arg_order": ['input', 'other'],
    # set limits based on index of argument in the order
    # three lists per argument: value list, dtype list, range list
    # min_num, max_num, min_size, max_size
    "limits":   [            
                    domain_limits['tensor'], # input value
                    domain_limits['tensor_dtype'],  # input dtype
                    domain_limits['tensor_value_range'], # input value range
                    domain_limits['tensor'], # other value
                    domain_limits['tensor_dtype'],  # other dtype
                    domain_limits['tensor_value_range'] # other value range
                ]
}

# define rule 4
r4_definition = {
    "signature":    {
                        "input": "tensor",
                        "other": "tensor"
                    },
    "ruleset":  set([
                    (2, 'rule_4', 'input', 'other')
                ]),
    "random_candidate": {
                            "input": np.random.rand(3,2,5,9,343,2).astype(np.int32),
                            "other": np.random.rand(4, 2).astype(np.complex128)
                        },
    "arg_order": ['input', 'other'],
    # set limits based on index of argument in the order
    # three lists per argument: value list, dtype list, range list
    # min_num, max_num, min_size, max_size
    "limits":   [            
                    domain_limits['tensor'], # input value
                    domain_limits['tensor_dtype'],  # input dtype
                    domain_limits['tensor_value_range'], # input value range
                    domain_limits['tensor'], # other value
                    domain_limits['tensor_dtype'],  # other dtype
                    domain_limits['tensor_value_range'] # other value range
                ]
}

# define rule 5
r5_definition = {
    "signature":    {
                        "input": "tensor",
                        "index": "tensor"
                    },
    "ruleset":  set([
                    (2, 'rule_5', 'input', 'index')
                ]),
    "random_candidate": {
                            "input": np.random.rand(3,2,5,9,343,2),
                            "index": np.random.rand(4, 2).astype(np.int64)
                            # Fixing the dtype for now, we might need a uniary invariant to make sure this stays int or update rule 5 to require index tensors to be of int types
                        },
    "arg_order": ['input', 'index'],
    # set limits based on index of argument in the order
    # three lists per argument: value list, dtype list, range list
    # min_num, max_num, min_size, max_size
    "limits":   [            
                    domain_limits['tensor'], # input value
                    domain_limits['tensor_dtype'],  # input dtype
                    domain_limits['tensor_value_range'], # input value range
                    domain_limits['tensor'], # index value
                    domain_limits['tensor_dtype'],  # index dtype
                    domain_limits['tensor_value_range'] # index value range
                ]
}

# define rule 6
r6_definition = {
    "signature":    {
                        "input": "tensor",
                        "other": "tensor"
                    },
    "ruleset":  set([
                    (2, 'rule_6', 'input', 'other')
                ]),
    "random_candidate": {
                            "input": np.random.rand(3,2,5,9,343,2),
                            "other": np.random.rand(4, 2)
                        },
    "arg_order": ['input', 'other'],
    # set limits based on index of argument in the order
    # three lists per argument: value list, dtype list, range list
    # min_num, max_num, min_size, max_size
    "limits":   [            
                    domain_limits['tensor'], # input value
                    domain_limits['tensor_dtype'],  # input dtype
                    domain_limits['tensor_value_range'], # input value range
                    domain_limits['tensor'], # other value
                    domain_limits['tensor_dtype'],  # other dtype
                    domain_limits['tensor_value_range'] # other value range
                ]
}

# define rule 11
r11_definition = {
    "signature":    {
                        "input": "tensor",
                        "dim": "integer",
                        "index": "tensor"
                    },
    "ruleset":  set([
                    (3, 'rule_11', 'input', 'dim', 'index')
                ]),
    "random_candidate": {
                            "input": np.random.rand(3,2,5,9,343,2),
                            "dim": 10,
                            "index": np.random.rand(4, 2)
                        },
    "arg_order": ['input', 'dim', 'index'],
    # set limits based on index of argument in the order
    # three lists per argument: value list, dtype list, range list
    # min_num, max_num, min_size, max_size
    "limits":   [            
                    domain_limits['tensor'], # input value
                    domain_limits['tensor_dtype'],  # input dtype
                    domain_limits['tensor_value_range'], # input value range
                    domain_limits['integer'], # dim value
                    domain_limits['integer_dtype'],  # dim dtype, integer only
                    domain_limits['integer_value_range'],    # dim value range, we don't use it for integers, it is here just for cohesion
                    domain_limits['tensor'], # index value
                    domain_limits['tensor_dtype'],  # index dtype
                    domain_limits['tensor_value_range'] # index value range
                ]
}

############### map definitions to rules/apis ################

map_defs = {
    "rule_1": r1_definition,
    "rule_2": r2_definition,
    "rule_3": r3_definition,
    "rule_4": r4_definition,
    "rule_5": r5_definition,
    "rule_6": r6_definition,
    "rule_11": r11_definition,
    "scatter": scatter_definition,
    "atan2": atan2_definition,
    "matmul": matmul_definition,
    "argmin": argmin_definition,
    "conv_transpose2d": conv_transpose2d_definition
}


def main():
    # Rules
    seed = 42
    for rule_name in ['rule_1','rule_2','rule_3','rule_4','rule_5','rule_6', 'rule_11']:
        start = time.time()
        # Set a random seed for reproducibility
        seed += 1
        config = Configuration(map_defs[rule_name], seed) # Rule configuration
        mutator = Mutator(config)
        (best_distance, best_input) = optimize(config, mutator)
        print(f"\n{' '.join(rule_name.capitalize().split('_'))}:\nBest Distance: {best_distance}\nBest Input:")
        best_input_abstracted = config.translate_to_input_dict(best_input, abstract=True)
        for key, value in best_input_abstracted.items():
            print(f"\t{key}: {value}")
        print(f"Optimized in {round(time.time()-start, 4)}s")
    
    
    # # Scatter
    # start = time.time()
    # seed = 200
    # map_defs["scatter"]["random_candidate"] =    {
    #                                                 "input": np.random.rand(2,4).astype(np.float32),
    #                                                 "dim": 7,
    #                                                 "index": np.random.rand(2,1).astype(np.int64),
    #                                                 "src": np.random.rand(2,1,3).astype(np.complex128)
    #                                             }
    # config = Configuration(map_defs["scatter"], seed) # Rule 2 and 3 configuration
    # mutator = Mutator(config)
    # (best_distance, best_input) = optimize(config, mutator)
    # print(f"\nScatter:\nBest Distance: {best_distance}\nBest Input:")
    # best_input_abstracted = config.translate_to_input_dict(best_input, abstract=True)
    # for key, value in best_input_abstracted.items():
    #     print(f"\t{key}: {value}")
    # print(f"Optimized in {round(time.time()-start, 4)}s")
    
    # # verify if the input is valid
    # try:
    #     scatter_cpu = get_driver("scatter")(config.translate_to_input_dict(best_input), cpu=True)
    #     print("\nThe input for scatter was valid! Yayyy!!!")
    # except Exception as e:
    #     print(f"\nThe input might be invalid. Faced exception:\n{e.__class__}: {str(e)}\n\n")
        
    #     ## Traceback for debugging
    #     traceback.print_exc()
    
def run_api_with_duration(api, duration, n_max=0, limit=30, print_details=False, lib="torch"):
    driver = get_driver(api)

    print(f"Optimizing for {api} with a {duration} second budget")
    execution_time = 0
    start = time.time()
    elapsed = 0
    valid = 0
    invalid = 0
    crash = 0
    excp = 0
    seed = 200
    generated_inputs = []
    definition = get_definition(api, lib=lib)
    if len(definition["ruleset"]) == 0:
        print(f"No invariants learned for {api}")
        return
    while elapsed < duration:
        seed += 1
        config = Configuration(definition, seed)
        # TODO: Debug why initializing random candidate makes optimizer slow
        config.random_candidate, _ = get_random_input(definition["signature"], config.rng, lib=lib)
        # config.set_random_candidate(map_defs[api]["random_candidate"])
        mutator = Mutator(config)
        (best_distance, best_input) = optimize(config, mutator, duration=limit)
        # Save abstract versions of the inputs with seed for reproduction
        generated_inputs.append((best_distance, best_input, seed))
        if print_details:
            print(f"\n{api}:\nBest Distance: {best_distance}\nBest Input:")
            best_input_abstracted = config.translate_to_input_dict(best_input, abstract=True)
            for key, value in best_input_abstracted.items():
                print(f"\t{key}: {value}")
        
        # verify if the input is valid
        start_execution = time.time()
        status, exception_message = oracle_crash(api, config.translate_to_input_dict(best_input, seed=seed), cpu=True, lib=lib)
        if status == "nominal":
            valid += 1
        elif status == "invalid":
            invalid += 1
            ## Traceback for debugging
            if print_details:
                print(f"\nThe input might be invalid. Faced exception:\n{exception_message}")
        elif status.endswith("_excp"):
            excp += 1
            # Always log crashes
            print(f"\n[{status}]\n{exception_message}")
            print(f"\nAbstract input: {best_input}")
        elif status.endswith("_crash"):
            crash += 1
            # Always log crashes
            print(f"\n[{status}]\n{exception_message}")
            print(f"\nAbstract input: {best_input}")
        else:
            if print_details:
                print(f"\nThe input faced status {status}. Faced exception:\n{exception_message}")
        execution_time = execution_time + time.time() - start_execution
        print(f"Valid: {valid} | Invalid: {invalid} | Crash: {crash} | Exception: {excp}", end='\r', flush=True)
        
        # If n_max is defined and n_max inputs have been generated, exit
        if n_max > 0 and (valid+invalid) == n_max:
            break
        
        elapsed = time.time() - start
    
    total_time = time.time() - start
    total = valid + invalid + crash
    valid_prcnt = round((valid+crash)*100/total,2) if total > 0 else 0
    print(f"\n[{api}]\n\tOptimzation took {round(total_time-execution_time, 4)}s\n\tExecuting {valid+invalid} inputs on {api} took {round(execution_time, 4)}s\n\tTotal {round(total_time, 4)}s")
    print(f"Valid: {valid} | Invalid: {invalid} | Crash: {crash} | Exception: {excp} | Total {total} | Validity Rate: {valid_prcnt}%")
    
    # Save outputs
    tmp_results = create_subdir(get_tmp_dir(), "fuzz_results")
    csv_file = os.path.join(tmp_results, f"{api}_{duration}.csv")
    with open(csv_file, "w") as f:
        # api, valid, invalid, crash, excp, total, valid_prcnt
        f.write(f"{api},{valid},{invalid},{crash},{excp},{total},{valid_prcnt}\n")
    input_dir = create_subdir(get_tmp_dir(), "fuzz_inputs")
    with open(os.path.join(input_dir, f"{api}_inputs.pkl"), "wb") as f_in:
        pickle.dump(generated_inputs, f_in)

if __name__ == "__main__":
    main()
    # Run scatter for 30 minutes
    duration = 30 # seconds
    limit = 10  # random restart after <limit> seconds
    print_details = str(sys.argv[1]).lower() == "true" if len(sys.argv) > 1 else False
    
    run_api_with_duration("scatter", duration, print_details=print_details, limit=limit)
    
    # Run atan2 for 30 seconds
    run_api_with_duration("atan2", duration, print_details=print_details, limit=limit)
    
    # Run argmin for 30 seconds
    run_api_with_duration("argmin", duration, print_details=print_details, limit=limit)
    
    # Run conv_transpose2d for 30 seconds
    run_api_with_duration("conv_transpose2d", duration, print_details=print_details, limit=limit)