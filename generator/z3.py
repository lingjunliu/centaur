import time
import numpy as np
from z3 import *
from .input_generators import get_ll, abstract_print
from .rules_auto_z3 import get_rules_map
from .definitions import get_definition
from .serialize import load_model, save_model
from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, MAX_SZ_TENSOR, list_of_available_dtypes, domain_limits, list_of_string_values
from utils.misc import create_subdir, get_tmp_dir, get_dir_in_root
from utils.new_api_utils import get_lib_version
from eval.oracle import oracle_crash
from functools import reduce
import os
import random
import json
from utils.proc import get_memory_usage

def save_state_models(api, suffix, unsat, nominal, invalid, crash, excp, tmp_results):
    api = f"{api}_{suffix}" if suffix > 0 else api
    total = nominal + invalid + crash + excp
    valid_prcnt = round((total-invalid)*100/total,2) if total > 0 else 0
    # Save outputs
    csv_file = os.path.join(tmp_results, f"{api}.csv")
    with open(csv_file, "w") as f:
        # api, unsat, nominal, invalid, crash, excp, total, valid_prcnt
        f.write(f"{api},{unsat},{nominal},{invalid},{crash},{excp},{total},{valid_prcnt}\n")

def create_z3_args(signature):
    z3_args = {}
    for param, typ in signature.items():
        if typ == "integer":
            z3_args[param] = {
                "value": Int(f"{param}_value"),
                "dtype": Int(f"{param}_dtype")
            }
        elif typ == "float":
            z3_args[param] = {
                "value": Real(f"{param}_value"),
                "dtype": Int(f"{param}_dtype")
            }
        elif typ == "boolean":
            z3_args[param] = Bool(param)
        elif typ == "string":
            z3_args[param] = {
                "value": Int(f"{param}_value"), # string is represented as an index in the list of string values
                "dtype": Int(f"{param}_dtype")
            }
        elif typ in ("tuple", "list"):
            z3_args[param] = {
                "length": Int(f"{param}_length"),
                "values": Array(f"{param}_values", IntSort(), IntSort())
            }
        elif typ == "tensor" or typ == "tensor_list":
            z3_args[param] = {
                "ndim": Int(f"{param}_ndim"),
                "shape": Array(f"{param}_shape", IntSort(), IntSort()),
                "dtype": Int(f"{param}_dtype"),
                "range": Array(f"{param}_range", IntSort(), IntSort())
            }
        elif typ == "dtype":
            z3_args[param] = Int(param)
        else:
            raise ValueError(f"Unsupported type: {typ}")
    return z3_args

def initial_constraints(solver, signature, z3_args):
    for param_name, z3_var in z3_args.items():
        param_type = signature[param_name]

        if param_type == "tensor" or param_type == "tensor_list":
            ndim, shape, dtype, range_ = z3_var['ndim'], z3_var['shape'], z3_var['dtype'], z3_var['range']
            solver.add(And(ndim >= 1, ndim <= MAX_N_DIM))
            solver.add(And(*[
                Implies(i < ndim, And(Select(shape, i) >= 0, Select(shape, i) <= MAX_SZ_DIM))
                for i in range(MAX_N_DIM)
            ]))
            solver.add(And(dtype >= 0, dtype <= len(list_of_available_dtypes) - 3))
        
            solver.add(And(*[
                Select(range_, 0) >= -MAX_SZ_NUM, Select(range_, 0) <= MAX_SZ_NUM,
                Select(range_, 1) >= -MAX_SZ_NUM, Select(range_, 1) <= MAX_SZ_NUM,
                Select(range_, 0) <= Select(range_, 1)
            ])) 
            size = reduce(lambda acc, i: acc * If(i < ndim, Select(shape, i), 1), range(MAX_N_DIM), 1)
            solver.add(size * 0.001 * 0.001 < MAX_SZ_TENSOR)

        elif param_type == "list" or param_type == "tuple":
            length, values = z3_var['length'], z3_var['values']

            solver.add(And(length >= 1, length <= MAX_N_DIM))
            solver.add(And(*[
                Implies(i < length, And(Select(values, i) >= -MAX_SZ_DIM, Select(values, i) <= MAX_SZ_DIM))
                for i in range(MAX_N_DIM)
            ]))
        
        elif param_type in ["integer", "float", "string"]:
            value, dtype = z3_var['value'], z3_var['dtype']
            solver.add(And(value >= domain_limits[f'{param_type}_value_range'][0], value <= domain_limits[f'{param_type}_value_range'][1]))
            solver.add(And(dtype >= domain_limits[f'{param_type}_dtype'][0], dtype <= domain_limits[f'{param_type}_dtype'][1]))
        elif param_type == "boolean":
            solver.add(Or(z3_var == domain_limits[f'{param_type}_value_range'][0], z3_var == domain_limits[f'{param_type}_value_range'][1]))            # Two possible values, True or False
        elif param_type == "dtype":
            solver.add(And(z3_var >= 0, z3_var <= len(list_of_available_dtypes) - 3)) 

def collect_constraints(solver, ruleset, z3_args, use_reference=False):
    rule_func_map = get_rules_map(use_reference=use_reference)
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

        if param_type == "tensor" or param_type == "tensor_list":
            ndim = model.eval(z3_var['ndim'], model_completion=True).as_long()
            shape = [model.eval(Select(z3_var['shape'], i), model_completion=True).as_long() for i in range(ndim)]
            dtype = model.eval(z3_var['dtype'], model_completion=True).as_long()
            low = model.eval(Select(z3_var['range'], 0), model_completion=True).as_long()
            high = model.eval(Select(z3_var['range'], 1), model_completion=True).as_long()
           
            np_array = np.random.uniform(low, high, size=shape).astype(list_of_available_dtypes[dtype])
            concrete_args[param_name] = np_array
            
        elif param_type == "list":
            length = model.eval(z3_var['length'], model_completion=True).as_long()
            values = z3_var['values']

            concrete_args[param_name] = [model.eval(Select(values, i), model_completion=True).as_long() for i in range(length)]

        elif param_type == "tuple":
            length = model.eval(z3_var['length'], model_completion=True).as_long()
            values = z3_var['values']

            concrete_args[param_name] = tuple([model.eval(Select(values, i), model_completion=True).as_long() for i in range(length)])
        elif param_type == "integer":
            value = model.eval(z3_var['value'], model_completion=True).as_long()
            dtype = model.eval(z3_var['dtype'], model_completion=True).as_long()
            concrete_args[param_name] = list_of_available_dtypes[dtype](value)
        elif param_type == "float":
            value = model.eval(z3_var['value'], model_completion=True).as_fraction()
            dtype = model.eval(z3_var['dtype'], model_completion=True).as_long()
            concrete_args[param_name] = list_of_available_dtypes[dtype](value.numerator/value.denominator)
        elif param_type == "string":
            value = model.eval(z3_var['value'], model_completion=True).as_long()
            dtype = model.eval(z3_var['dtype'], model_completion=True).as_long()
            concrete_args[param_name] = list_of_available_dtypes[dtype](list_of_string_values[value])
        elif param_type == "dtype":
            dtype = model.eval(z3_var, model_completion=True).as_long()
            concrete_args[param_name] = list_of_available_dtypes[dtype]
        else:
            value = model.eval(z3_var, model_completion=True)
            if isinstance(value, BoolRef):
                concrete_args[param_name] = random.choice([True, False]) 
            else:
                concrete_args[param_name] = value

        abstract_args[param_name] = get_ll(param_type, concrete_args[param_name])

    return concrete_args, abstract_args

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

# Getting the minimum and maximum values that Z3 variables can have 
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
        opt_min = Optimize()
        opt_min.add(linear_assertions)
        opt_min.minimize(var)
        if opt_min.check() == sat:
            val = opt_min.model().eval(var, model_completion=True)
            minv = float(val.as_fraction()) if sort_kind == Z3_REAL_SORT else val.as_long()
        else:
            continue
        opt_max = Optimize()
        opt_max.add(linear_assertions)
        opt_max.maximize(var)
        if opt_max.check() == sat:
            val = opt_max.model().eval(var, model_completion=True)
            maxv = float(val.as_fraction()) if sort_kind == Z3_REAL_SORT else val.as_long()
        else:
            continue
        bounds[var] = set([minv, maxv]) 
    return bounds

# Add assertions for sampled values from partitions
def sample_partitions(var_values_map, p):
    sampled_partitions = set()
    all_vars = list(var_values_map.keys())

    sample_size = int(len(all_vars) * p)
    sampled_vars = random.sample(all_vars, sample_size)

    for var in sampled_vars:
        values = sorted(var_values_map[var])
        if len(values) < 2:
            continue

        idx = random.randint(0, len(values) - 2)
        v1, v2 = values[idx], values[idx + 1]

        sort_kind = var.sort().kind()
        if sort_kind == Z3_INT_SORT:
            if int(v2) - int(v1) <= 1:
                continue 
            v = random.randint(int(v1) + 1, int(v2) - 1)
        elif sort_kind == Z3_REAL_SORT:
            if abs(v2 - v1) <= 1e-6:
                continue
            v = random.uniform(v1 + 1e-6, v2 - 1e-6)
        else:
            continue

        assertion = (var == v)
        sampled_partitions.add(assertion)

    return sampled_partitions

def gen_models(definition, api, z3_args, model_gen_duration, max_model=0, seed=42, print_details=False, saturation=10, lib="torch", corpus_dir=None, return_models=True, use_reference=False):
    elapsed = 0
    start = time.time()

    # initialization
    solver = Solver()
    models, num_model = [], 0
    initial_constraints(solver, definition["signature"], z3_args)
    collect_constraints(solver, definition["ruleset"], z3_args, use_reference=use_reference)
    block_all = set()
    stale = 0
    # valid_blocks = []   # list of blocks for valid models, saved for restarts
    rng = np.random.default_rng(seed)

    nominal = 0
    invalid = 0
    crash = 0
    excp = 0
    unsat = 0

    tmp_results = create_subdir(get_tmp_dir(), "model_results")
    var_values_map = variable_bounds(solver.assertions())

    while elapsed < model_gen_duration and (num_model < max_model or max_model == 0):
        block_one = []
        one_solver = Solver()
        one_solver.add(*solver.assertions())

        # Strategy #1: Adding blocking constraints with probability p_1
        sampled_blocks = random.sample(list(block_all), int(len(block_all) * 0.3))
        one_solver.add(*sampled_blocks)

        # Strategy #2: Adding partitioning-based constraints with probability p_2
        sampled_partitions = sample_partitions(var_values_map, 0.3)
        one_solver.add(*sampled_partitions)
        
        if one_solver.check() != sat:
            unsat += 1
            if stale > saturation:
                # restart the solver
                solver = Solver()
                initial_constraints(solver, definition["signature"], z3_args)
                collect_constraints(solver, definition["ruleset"], z3_args, use_reference=use_reference)
                # solver.add(And(valid_blocks))   # Adding previously saved blocks from valid models
                # block = []
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
        for decl in model.decls():
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
                    # potential_valid_blocks.append(Select(var, i) != model.eval(Select(var, i), model_completion=True))
                    # Do not dim_size to be 0 more than once for a dimension in the shape
                    # if model.eval(Select(var, i), model_completion=True).as_long() == 0 and suffix == "shape":
                    #     solver.add(Select(var, i) != model.eval(Select(var, i), model_completion=True))
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
                # if not suffix and suffix not in ["ndim", "dtype"]: # potentially can add length too, TODO: asess
                #     potential_valid_blocks.append(var != val)
        
        # Randomly block one of the constraints
        # selected_const = block[rng.integers(len(block))]
        # solver.add(selected_const)
        # if print_details:
        #     print(f"Blocking constraint: {selected_const}")

        # For strategy #1: The set of blocking constraints is updated
        for elem in block_one:
            if elem not in block_all:
                block_all.add(elem)

        concrete_input, abstract_input = instantiate_args(model, definition["signature"], z3_args)
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
            
            print(f"Valid models: {num_model} | Memory usage: {get_memory_usage():.4f} MB")
            # TODO: Check if this could be improved
            # selected_valid_block = potential_valid_blocks[rng.integers(len(potential_valid_blocks))]
            # solver.add(selected_valid_block)
            # valid_blocks.append(selected_valid_block)
            if status != "nominal": # Always log crashes
                print(f"\n[{status}]\n{exception_message}")
                print(f"\nPotential bug. Input:\n{abstract_print(abstract_input, definition['signature'])}")
        elif print_details:
            invalid += 1
            print(abstract_print(abstract_input, definition["signature"]))
            print(f"\nThe input faced status {status}. Faced exception:\n{exception_message}")
        
        elapsed = time.time() - start

        save_state_models(definition["api"], definition["suffix"], unsat, nominal, invalid, crash, excp, tmp_results)

    print(f"Generated {num_model} models for {api}")
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

def run_model_gen(api, duration, n_max, lib, seed, regen, use_reference=False):
    print_details = False # Set to True if you want to print details of the process
    
    # alias
    if lib == "tensorflow":
        lib = "tf"
    elif lib == "pytorch":
        lib = "torch"

    # Check if it is a variation of the API
    if "_" in api:
        api, suffix = api.rsplit("_", 1)
        if suffix.isdigit():
            suffix = int(suffix)
        else:
            suffix = 0
            api = f"{api}_{suffix}"  # Reconstruct the API name with suffix
    else:
        suffix = 0
    
    api = get_lib_version(api, lib=lib)
    definition = get_definition(api, z3=True, lib=lib, suffix=suffix, use_reference=use_reference)
    if len(definition["ruleset"]) == 0:
        print(f"No invariants learned for {api}")
        return
    
    corpus_dir = "corpus_tf" if lib == "tf" else "corpus_torch"
    corpus_dir = os.path.join(get_dir_in_root(corpus_dir), f"{api}_{suffix}" if suffix > 0 else api)
    z3_args = create_z3_args(definition["signature"])
    if os.path.exists(corpus_dir) and not regen:
        models = load_existing_models(corpus_dir, z3_args)
        print(f"Loaded {len(models)} existing models for {api}")
    else:
        os.makedirs(corpus_dir, exist_ok=True)
        models = gen_models(definition, api, z3_args, duration, max_model=n_max, seed=seed, print_details=print_details, corpus_dir=corpus_dir, return_models=False, use_reference=use_reference)
    

def main():
    if len(sys.argv) < 3:
        print("Usage: python fuzz.py <api> <duration> <lib, default='torch'> <seed, optional> <n_max, optional> <regen, default=False>")
        return
    
    api = sys.argv[1]
    duration = int(sys.argv[2])
    n_max = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    lib = sys.argv[4] if len(sys.argv) > 4 else "torch"
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    regen = int(sys.argv[6]) == 1 if len(sys.argv) > 6 else False
    use_reference = int(sys.argv[7]) == 1 if len(sys.argv) > 7 else False
    
    run_model_gen(api, duration, n_max, lib, seed, regen, use_reference=use_reference)

if __name__ == "__main__":
    main()