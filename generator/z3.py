import time
import numpy as np
from z3 import *
from .input_generators import abstract_print
from .definitions import get_definition
from .serialize import load_model, save_model
from utils.defaults import MAX_N_DIM, int_buckets, float_buckets
from utils.misc import create_subdir, get_tmp_dir, get_dir_in_root, bcolors
from utils.new_api_utils import get_lib_version, get_api_suffix
from utils.z3_utils import instantiate_args, create_z3_args, initial_constraints, collect_constraints, parition_solvers, add_negative_buckets
from eval.oracle import oracle_crash
import os
import json
from utils.proc import get_memory_usage
import shutil

def save_state_models(api, suffix, unsat, nominal, invalid, crash, excp, tmp_results):
    api = f"{api}_{suffix}" if suffix > 0 else api
    total = nominal + invalid + crash + excp
    valid_prcnt = round((total-invalid)*100/total,2) if total > 0 else 0
    # Save outputs
    csv_file = os.path.join(tmp_results, f"{api}.csv")
    with open(csv_file, "w") as f:
        # api, unsat, nominal, invalid, crash, excp, total, valid_prcnt
        f.write(f"{api},{unsat},{nominal},{invalid},{crash},{excp},{total},{valid_prcnt}\n")

def is_const_num(expr):
    return is_int_value(expr) or is_rational_value(expr)

def is_nonlinear_expr(expr):
    kind = expr.decl().kind()
    nonlinear_kinds = {Z3_OP_MUL, Z3_OP_DIV, Z3_OP_POWER}
    
    if kind in nonlinear_kinds:
        children = expr.children()
        non_const_children = [c for c in children if not is_const_num(c)]
        if len(non_const_children) >= 2:
            return True

    for c in expr.children():
        if is_nonlinear_expr(c):
            return True
    return False

# Z3 Optimize().{maximize, minimize} cannot handle nonlinear assertions
def is_nonlinear_assertion(assertion):
    if assertion.decl().kind() in [Z3_OP_AND, Z3_OP_OR, Z3_OP_IMPLIES]:
        return any(is_nonlinear_assertion(c) for c in assertion.children())
    return is_nonlinear_expr(assertion)

def clip_buckets(buckets, min_val, max_val):
    """
    Clip the buckets to the specified range [min_val, max_val].
    """
    return sorted([min_val] + [b for b in buckets if min_val < b < max_val] + [max_val])

# Getting the minimum and maximum values that Z3 variables can have 
# Then adding buckets within the range to sample from
def variable_bounds(assertions):
    def collect_vars(expr):
        vars_found = set()
        def walk(e):
            if is_const(e) and e.decl().kind() == Z3_OP_UNINTERPRETED:
                if e.sort().kind() != Z3_ARRAY_SORT and e.sort().kind() != Z3_BOOL_SORT:
                    vars_found.add(e)
            elif e.decl().kind() == Z3_OP_SELECT:
                arr, idx = e.children()
                if is_const(arr) and is_int_value(idx):
                    vars_found.add(Select(arr, idx))
            for ch in e.children():
                walk(ch)
        walk(expr)
        return vars_found

    all_vars = set()
    for a in assertions:
        all_vars |= collect_vars(a)

    bounds = {}
    linear_assertions = [a for a in assertions if not is_nonlinear_assertion(a)]

    for var in all_vars:
        sort_kind = var.sort().kind()
        if sort_kind == Z3_REAL_SORT:
            default_buckets = float_buckets
        else:
            default_buckets = int_buckets
        
        default_buckets = add_negative_buckets(default_buckets)
        
        opt_min = Optimize()
        opt_min.set("timeout", 1000)
        opt_min.add(linear_assertions)
        opt_min.minimize(var)
        if opt_min.check() == sat:
            val = opt_min.model().eval(var, model_completion=True)
            minv = float(val.as_fraction()) if sort_kind == Z3_REAL_SORT else val.as_long()
        else:
            continue
        opt_max = Optimize()
        opt_max.set("timeout", 1000)
        opt_max.add(linear_assertions)
        opt_max.maximize(var)
        if opt_max.check() == sat:
            val = opt_max.model().eval(var, model_completion=True)
            maxv = float(val.as_fraction()) if sort_kind == Z3_REAL_SORT else val.as_long()
        else:
            continue
        bounds[var] = set(clip_buckets(default_buckets, minv, maxv)) if minv != maxv else set([minv])
    return bounds

# Add assertions for sampled values from partitions
def sample_partitions(var_values_map, p, rng=np.random.default_rng(42)):
    sampled_partitions = set()
    all_vars = list(var_values_map.keys())

    sample_size = int(len(all_vars) * p)
    sampled_vars = rng.choice(all_vars, sample_size, replace=False)

    for var in sampled_vars:
        values = sorted(var_values_map[var])
        if len(values) < 2:
            continue

        idx = rng.integers(0, len(values) - 1)
        v1, v2 = values[idx], values[idx + 1]

        sort_kind = var.sort().kind()
        if sort_kind == Z3_INT_SORT:
            if int(v2) - int(v1) < 2:
                continue 
            v = rng.integers(int(v1) + 1, int(v2))
        elif sort_kind == Z3_REAL_SORT:
            if abs(v2 - v1) <= 1e-6:
                continue
            v = rng.uniform(v1 + 1e-6, v2 - 1e-6)
        elif sort_kind == Z3_BOOL_SORT:
            v = rng.choice([False, True])
        else:
            continue

        assertion = (var == v)
        sampled_partitions.add(assertion)

    return sampled_partitions

def gen_models(definition, api, z3_args, model_gen_duration, max_model=0, seed=42, print_details=False, saturation=10, lib="torch", corpus_dir=None, return_models=True, use_reference=False):
    elapsed = 0
    start = time.time()

    # initialization
    solver_main = Solver()
    models, num_model = [], 0
    initial_constraints(solver_main, definition["signature"], z3_args, lib=lib)
    collect_constraints(solver_main, api, definition["ruleset"], z3_args, use_reference=use_reference, lib=lib)
    block_all = set()
    perma_block = set()
    stale = 0
    # valid_blocks = []   # list of blocks for valid models, saved for restarts
    rng = np.random.default_rng(seed)

    nominal = 0
    invalid = 0
    crash = 0
    excp = 0
    unsat = 0

    tmp_results = create_subdir(get_tmp_dir(), "model_results")
    var_values_map = variable_bounds(solver_main.assertions())

    solve_times = []
    check_times = []
    start_time = time.time()
    while elapsed < model_gen_duration and (num_model < max_model or max_model == 0):
        # Strategy #3: Partitioning the solver for boolean variables and add different ranges
        partitioned_solvers = parition_solvers(solver_main, definition["signature"], z3_args, lib=lib)
        for solver in partitioned_solvers:
            block_one = []
            one_solver = Solver()
            one_solver.add(*solver.assertions())

            # Strategy #1: Adding blocking constraints with probability p_1
            sampled_blocks = rng.choice(list(block_all), int(len(block_all) * 0.3), replace=False)
            one_solver.add(*sampled_blocks)
            one_solver.add(*perma_block)

            # Strategy #2: Adding partitioning-based constraints with probability p_2
            sampled_partitions = sample_partitions(var_values_map, 0.3)
            one_solver.add(*sampled_partitions)
            
            if one_solver.check() != sat:
                unsat += 1
                if stale > saturation:
                    # restart the solver
                    solver_main = Solver()
                    initial_constraints(solver_main, definition["signature"], z3_args, lib=lib)
                    collect_constraints(solver_main, api, definition["ruleset"], z3_args, use_reference=use_reference, lib=lib)
                    stale = 0
                    seed += 1
                    saturation += 10    # Making it more difficult to reach stale
                    rng = np.random.default_rng(seed)
                    continue

                stale += 1
                elapsed = time.time() - start
                continue
            
            # potential_valid_blocks = []
            model = one_solver.model()
            solve_times.append(time.time() - start_time)
            start_time = time.time()
            for decl in model.decls():
                if decl.arity() != 0:
                    continue
                var, val = decl(), model[decl]
                name_parts = str(decl.name()).rsplit("_", 1)

                if len(name_parts) == 2:
                    prefix, suffix = name_parts
                else:
                    prefix, suffix = name_parts[0], None

                if val.sort().kind() == Z3_ARRAY_SORT:                
                    array_len = None
                    if suffix == "shape" or suffix == "values":
                        for other_decl in model.decls():
                            if str(other_decl.name()) in [f"{prefix}_ndim", f"{prefix}_length"]:
                                array_len = model.eval(other_decl(), model_completion=True).as_long()
                        if array_len is None:
                            array_len = MAX_N_DIM
                    elif suffix == "range":
                        array_len = 2
                    for i in range(array_len):
                        block_one.append(Select(var, i) != model.eval(Select(var, i), model_completion=True))
                        # For strategy #2: Value set which a partition is created from is updated 
                        actual_key = None
                        for key in var_values_map.keys():
                            if key.sexpr() == Select(var, i).sexpr():
                                actual_key = key
                                break
                        if actual_key is not None:
                            var_values_map[actual_key].add(model.eval(Select(var, i), model_completion=True).as_long())
                        
                        # Do not dim_size to be 0 more than once for a dimension in the shape
                        if model.eval(Select(var, i), model_completion=True).as_long() == 0 and suffix == "shape":
                            perma_block.add(Select(var, i) != model.eval(Select(var, i), model_completion=True))
                else:
                    block_one.append(var != val)
                    # For strategy #2: Value set which a partition is created from is updated 
                    actual_key = None
                    for key in var_values_map.keys():
                        if key.sexpr() == var.sexpr():
                            actual_key = key
                            break
                    if actual_key is not None:
                        val = model.eval(var, model_completion=True)
                        if var.sort().kind() == Z3_INT_SORT:
                            var_values_map[actual_key].add(val.as_long())
                        elif var.sort().kind() == Z3_REAL_SORT:
                            var_values_map[actual_key].add(float(val.as_fraction()))
                    
                    if suffix == "ndim" and val.as_long() == 0:
                        perma_block.add(var != val)

            # For strategy #1: The set of blocking constraints is updated
            for elem in block_one:
                if elem not in block_all:
                    block_all.add(elem)

            concrete_input, abstract_input = instantiate_args(model, definition["signature"], z3_args, lib=lib)
            status, exception_message = oracle_crash(api, concrete_input, cpu=True, lib=lib)
    
            if status != "invalid":
                if status == "cpu_crash":
                    crash += 1
                elif status == "cpu_excp":
                    excp += 1
                elif status == "nominal":
                    nominal += 1
                
                if return_models:
                    models.append(model)
                # Save the model
                if corpus_dir:
                    path = os.path.join(corpus_dir, f"model-{num_model}.json")
                    save_model(model, path)
                num_model += 1
                
                print(f"Valid models: {num_model} | Memory usage: {get_memory_usage():.4f} MB", end="\r", flush=True)

                if status != "nominal": # Always log crashes
                    print(f"\n[{status}]\n{exception_message}")
                    print(f"\nPotential bug. Input:\n{abstract_print(abstract_input, definition['signature'])}")
            else:
                invalid += 1
                if print_details:
                    print(abstract_print(abstract_input, definition["signature"]))
                    print(f"\nThe input faced status {status}. Faced exception:\n{exception_message}")
            
            elapsed = time.time() - start
            check_times.append(time.time() - start_time)
            start_time = time.time()

            save_state_models(definition["api"], definition["suffix"], unsat, nominal, invalid, crash, excp, tmp_results)

    print(f"Generated {num_model} models for {api} with suffix {definition['suffix']} | Avg solve time: {np.mean(solve_times):.2f} | Avg check time: {np.mean(check_times):.2f} s ")
    return models

def load_existing_models(corpus_dir, z3_args):
    models = []
    for model_file in sorted(os.listdir(corpus_dir)):
        model_path = os.path.join(corpus_dir, model_file)
        with open(model_path, "r") as f:
            model_data = json.load(f)
        model = load_model(model_data, z3_args)
        models.append(model)

    return models

def run_model_gen(variant, duration, n_max, lib, seed, regen, use_reference=False):
    print_details = False # Set to True if you want to print details of the process
    
    # alias
    if lib == "tensorflow":
        lib = "tf"
    elif lib == "pytorch":
        lib = "torch"

    # Check if it is a variation of the API
    api, suffix = get_api_suffix(variant)

    api = get_lib_version(api, lib=lib)
    definition = get_definition(api, z3=True, lib=lib, suffix=suffix, use_reference=use_reference)
    if len(definition["ruleset"]) == 0:
        print(f"No invariants learned for {api}")
        return
    else:
        print('-----' * 20)
        print(f"Using these rulesets for {api} with suffix {suffix}:")
        for arity, rule_name, *args in definition["ruleset"]:
            print(f"- {rule_name},{arity},{args}")
        print('-----' * 20)
    
    corpus_dir = "corpus_tf" if lib == "tf" else "corpus_torch"
    corpus_dir = os.path.join(get_dir_in_root(corpus_dir), f"{api}_{suffix}" if suffix > 0 else api)
    z3_args = create_z3_args(definition["signature"])
    if os.path.exists(corpus_dir) and not regen:
        models = load_existing_models(corpus_dir, z3_args)
        print(f"Loaded {len(models)} existing models for {api}")
    else:
        if os.path.exists(corpus_dir):
            print(f"Removing existing corpus directory: {corpus_dir}")
            shutil.rmtree(corpus_dir)
        os.makedirs(corpus_dir, exist_ok=True)
        start_time = time.time()
        models = gen_models(definition, api, z3_args, duration, max_model=n_max, seed=seed, print_details=print_details, corpus_dir=corpus_dir, return_models=False, use_reference=use_reference, lib=lib)
        print(f"{bcolors.OKBLUE}Model generation took {time.time()-start_time} s{bcolors.ENDC}")
    

def main():
    if len(sys.argv) < 3:
        print("Usage: python fuzz.py <variant> <duration> <lib, default='torch'> <seed, optional> <n_max, optional> <regen, default=False>")
        return
    
    variant = sys.argv[1]
    duration = int(sys.argv[2])
    n_max = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    lib = sys.argv[4] if len(sys.argv) > 4 else "torch"
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    regen = int(sys.argv[6]) == 1 if len(sys.argv) > 6 else False
    use_reference = int(sys.argv[7]) == 1 if len(sys.argv) > 7 else False
    
    run_model_gen(variant, duration, n_max, lib, seed, regen, use_reference=use_reference)

if __name__ == "__main__":
    main()
