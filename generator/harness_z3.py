import time
import numpy as np
import traceback
import os
import pickle
import sys

from z3 import *
from .ea import Configuration, Mutator, optimize
from .definitions import map_defs, get_definition
from .input_generators import get_ll
from .rules_z3 import rule_func_map
from utils.api_utils import get_driver
from utils.misc import create_subdir, get_tmp_dir, get_dir_in_root, read_pkl, save_to_new_pkl
from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, MAX_SZ_TENSOR, list_of_available_dtypes
from eval.oracle import oracle_crash
from functools import reduce

def create_z3_args(signature):
    z3_args = {}
    for param, typ in signature.items():
        if typ == "integer":
            z3_args[param] = Int(param)
        elif typ == "float":
            z3_args[param] = Real(param)
        elif typ == "boolean":
            z3_args[param] = Bool(param)
        elif typ == "string":
            z3_args[param] = String(param)
        elif typ in ("tuple", "list"):
            z3_args[param] = {
                "length": Int(f"{param}_length"),
                "values": Array(f"{param}_values", IntSort(), IntSort())
            }
        elif typ == "tensor":
            z3_args[param] = {
                "ndim": Int(f"{param}_ndim"),
                "shape": Array(f"{param}_shape", IntSort(), IntSort()),
                "dtype": Int(f"{param}_dtype"),
                "range": Array(f"{param}_range", IntSort(), IntSort())
            }
        else:
            raise ValueError(f"Unsupported type: {typ}")
    return z3_args

def initial_constraints(solver, signature, z3_args):
    for param_name, z3_var in z3_args.items():
        param_type = signature[param_name]

        if param_type == "tensor":
            ndim, shape, dtype, range_ = z3_var['ndim'], z3_var['shape'], z3_var['dtype'], z3_var['range']
            solver.add(And(ndim >= 1, ndim <= MAX_N_DIM))
            solver.add(And(*[
                Implies(i < ndim, And(Select(shape, i) >= 0, Select(shape, i) <= MAX_SZ_NUM))
                for i in range(MAX_N_DIM)
            ]))
            solver.add(And(dtype >= 0, dtype <= len(list_of_available_dtypes) - 3))
        
            solver.add(And(*[
                Select(range_, 0) >= -MAX_SZ_NUM, Select(range_, 0) <= MAX_SZ_NUM,
                Select(range_, 1) >= -MAX_SZ_NUM, Select(range_, 1) <= MAX_SZ_NUM,
                Select(range_, 0) < Select(range_, 1)
            ])) 
            size = reduce(lambda acc, i: acc * If(i < ndim, Select(shape, i), 1), range(MAX_N_DIM), 1)
            solver.add(size * 0.001 * 0.001 < MAX_SZ_TENSOR)

        elif param_type == "list" or param_type == "tuple":
            length, values = z3_var['length'], z3_var['values']

            solver.add(And(length >= 1, length <= MAX_N_DIM))
            solver.add(And(*[
                Implies(i < length, And(Select(values, i) >= -MAX_SZ_NUM, Select(values, i) <= MAX_SZ_NUM))
                for i in range(MAX_N_DIM)
            ]))

def collect_constraints(solver, ruleset, z3_args):
    for rule in ruleset:
        arity, rule_name, *args = rule
        rule_func = rule_func_map[arity][rule_name]
        
        arg_dicts = []
        for param_name in args:
            arg_dicts.append({param_name: z3_args[param_name]})
       
        rule_func(*arg_dicts, solver)

def instantiate_args(model, signature, z3_args, seed=42):
    concrete_args = {}
    abstract_args = {}
    rng = np.random.default_rng(seed)

    for param_name, z3_var in z3_args.items():
        param_type = signature[param_name]

        if param_type == "tensor":
            ndim = model.eval(z3_var['ndim']).as_long()
            shape = [model.eval(Select(z3_var['shape'], i)).as_long() for i in range(ndim)]
            dtype = model.eval(z3_var['dtype']).as_long()
            low = model.eval(Select(z3_var['range'], 0)).as_long()
            high = model.eval(Select(z3_var['range'], 1)).as_long()
        
            np_array = rng.uniform(low, high, size=shape).astype(list_of_available_dtypes[dtype])
            concrete_args[param_name] = np_array
            
        elif param_type == "list":
            length = model.eval(z3_var['length']).as_long()
            values = z3_var['values']

            concrete_args[param_name] = [model.eval(Select(values, i)).as_long() for i in range(length)]

        elif param_type == "tuple":
            length = model.eval(z3_var['length']).as_long()
            values = z3_var['values']

            concrete_args[param_name] = tuple([model.eval(Select(values, i)).as_long() for i in range(length)])

        else:
            value = model.eval(z3_var)
            if isinstance(value, IntNumRef):
                concrete_args[param_name] = value.as_long()
            elif isinstance(value, BoolRef):
                concrete_args[param_name] = is_true(value)
            elif isinstance(value, SeqRef):
                concrete_args[param_name] = value.as_string()
            else:
                concrete_args[param_name] = value
                
        abstract_args[param_name] = get_ll(param_type, concrete_args[param_name])

    return concrete_args, abstract_args
    
def gen_models(definition, driver, z3_args, model_gen_duration, max_model=0):
    elapsed = 0
    start = time.time()

    all_solver = Solver()
    models, num_model = [], 0

    while elapsed < model_gen_duration and (max_model == 0 or num_model < max_model):
        one_solver = all_solver.translate(all_solver.ctx)
        initial_constraints(one_solver, definition["signature"], z3_args)
        collect_constraints(one_solver, definition["ruleset"], z3_args)               
        
        if one_solver.check() != sat:
            elapsed = time.time() - start
            continue

        model = one_solver.model()
        block = []
        for decl in model.decls():
            var, val = decl(), model[decl]
            if val.sort().kind() == Z3_ARRAY_SORT:
                prefix = str(decl.name()).rsplit("_", 1)[0] 
                ndim = None
                for other_decl in model.decls():
                    if str(other_decl.name()) in [f"{prefix}_ndim", f"{prefix}_height"]:
                        ndim = model.eval(other_decl()).as_long()
                if ndim is None:
                    ndim = MAX_N_DIM
                for i in range(ndim):
                    block.append(Select(var, i) != model.eval(Select(var, i)))
            else:
                block.append(var != val)
        all_solver.add(Or(block))

        concrete_input, abstract_input = instantiate_args(model, definition["signature"], z3_args)
        status, exception_message = oracle_crash(driver, concrete_input, cpu=True)

        if status != "invalid":
            models.append(model)
            num_model += 1
            print(f"Valid models: {num_model}", end='\r', flush=True)
        
        elapsed = time.time() - start

    print(f"\nModel generation completed with {num_model} models")
    return models

def run_api_with_duration(api, model_gen_duration, fuzz_duration, max_model, n_max=0, print_details=False, model_regen=False):
    driver = get_driver(api)

    print(f"Optimizing for {api} with {model_gen_duration} (max_model) and {fuzz_duration} (fuzz) second budgets")
    execution_time = 0
    elapsed = 0
    valid = 0
    invalid = 0
    crash = 0
    seed = 200
    generated_inputs = []
    definition = get_definition(api, z3=True)
    if len(definition["ruleset"]) == 0:
        print(f"No invariants learned for {api}")
        return

    z3_args = create_z3_args(definition["signature"])
    
    models_file = os.path.join(get_dir_in_root("models"), f"{api}.pkl")
    if not os.path.exists(models_file) or model_regen:
        models = gen_models(definition, driver, z3_args, model_gen_duration, max_model)
        if len(models) == 0:
            print(f"No models generated for {api} within the {model_gen_duration} second time budget")
            return
        # save models to a file
        save_to_new_pkl(models_file, models)
    else:
        # read from saved models
        models = read_pkl(models_file)
    
    rng_model = np.random.default_rng(seed) # random generator for models

    start = time.time()
    while elapsed < fuzz_duration:
        seed += 1
        model = models[rng_model.integers(len(models))]
        concrete_input, abstract_input = instantiate_args(model, definition["signature"], z3_args, seed=seed)
        generated_inputs.append((0, abstract_input, seed))  # first element is distance, set as 0 for consistency

        start_execution = time.time()
        status, exception_message = oracle_crash(driver, concrete_input, cpu=True)
        if status == "nominal":
            valid += 1
        elif status == "invalid":
            invalid += 1
            ## Traceback for debugging
            if print_details:
                print(f"\nThe input might be invalid. Faced exception:\n{exception_message}")
        elif status == "cpu_crash":
            crash += 1
            # Always log crashes
            print(f"\n[CRASH]\n{exception_message}")
            print(f"\nAbstract input: {abstract_input}")
        else:
            if print_details:
                print(f"\nThe input faced status {status}. Faced exception:\n{exception_message}")
        execution_time = execution_time + time.time() - start_execution
        print(f"Valid: {valid} | Invalid: {invalid} | Crash: {crash}", end='\r', flush=True)

        # If n_max is defined and n_max inputs have been generated, exit
        if n_max > 0 and (valid+invalid) == n_max:
            break

        elapsed = time.time() - start

    total_time = time.time() - start
    total = valid + invalid + crash
    valid_prcnt = round((valid+crash)*100/total,2) if total > 0 else 0
    print(f"\n[{api}]\n\tOptimzation took {round(total_time-execution_time, 4)}s\n\tExecuting {valid+invalid} inputs on {api} took {round(execution_time, 4)}s\n\tTotal {round(total_time, 4)}s")
    print(f"Valid: {valid} | Invalid: {invalid} | Crash: {crash} | Total {total} | Validity Rate: {valid_prcnt}%")
    
    # Save outputs
    tmp_results = create_subdir(get_tmp_dir(), "fuzz_results")
    csv_file = os.path.join(tmp_results, f"{api}_{model_gen_duration}_{fuzz_duration}.csv")
    with open(csv_file, "w") as f:
        # api, valid, invalid, crash, total, valid_prcnt
        f.write(f"{api},{valid},{invalid},{crash},{total},{valid_prcnt}\n")
    input_dir = create_subdir(get_tmp_dir(), "fuzz_inputs")
    with open(os.path.join(input_dir, f"{api}_inputs.pkl"), "wb") as f_in:
        pickle.dump(generated_inputs, f_in)
        

if __name__ == "__main__":
    # Run scatter for 30 minutes
    model_gen_duration = 60 # seconds
    fuzz_duration = 30 # seconds
    max_model = 100
    print_details = sys.argv[1].lower() == 'true' if len(sys.argv) > 1 else False
    
    run_api_with_duration("kthvalue", model_gen_duration, fuzz_duration, max_model, print_details=print_details)
     
    # # Run atan2 for 30 seconds
    # run_api_with_duration("atan2", model_gen_duration, fuzz_duration, max_model, print_details=print_details)
    
    # # Run argmin for 30 seconds
    # run_api_with_duration("argmin", model_gen_duration, fuzz_duration, max_model, print_details=print_details)
    
    # # Run conv_transpose2d for 30 seconds
    # run_api_with_duration("conv_transpose2d", model_gen_duration, fuzz_duration, max_model, print_details=print_details)
