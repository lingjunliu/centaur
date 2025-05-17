import time
import numpy as np
from z3 import *
from .input_generators import get_ll, abstract_print
from .rules_z3 import rule_func_map
from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, MAX_SZ_TENSOR, list_of_available_dtypes, domain_limits, list_of_string_values
from eval.oracle import oracle_crash
from functools import reduce

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
                Implies(i < ndim, And(Select(shape, i) >= 0, Select(shape, i) <= MAX_SZ_NUM))
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
                Implies(i < length, And(Select(values, i) >= -MAX_SZ_NUM, Select(values, i) <= MAX_SZ_NUM))
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
                concrete_args[param_name] = is_true(value)
            else:
                concrete_args[param_name] = value

        abstract_args[param_name] = get_ll(param_type, concrete_args[param_name])

    return concrete_args, abstract_args
    
def gen_models(definition, driver, z3_args, model_gen_duration, max_model=0, seed=42, print_details=False, saturation=10):
    elapsed = 0
    start = time.time()

    # initialization
    solver = Solver()
    models, num_model = [], 0
    initial_constraints(solver, definition["signature"], z3_args)
    collect_constraints(solver, definition["ruleset"], z3_args)
    block = []
    stale = 0
    valid_blocks = []   # list of blocks for valid models, saved for restarts
    rng = np.random.default_rng(seed)

    while elapsed < model_gen_duration and (num_model < max_model or max_model == 0):
        if solver.check() != sat:
            if stale > saturation:
                # restart the solver
                solver = Solver()
                initial_constraints(solver, definition["signature"], z3_args)
                collect_constraints(solver, definition["ruleset"], z3_args)
                solver.add(And(valid_blocks))   # Adding previously saved blocks from valid models
                block = []
                stale = 0
                seed += 1
                rng = np.random.default_rng(seed)
                continue

            stale += 1
            elapsed = time.time() - start
            continue
        
        potential_valid_blocks = []
        model = solver.model()
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
                    block.append(Select(var, i) != model.eval(Select(var, i), model_completion=True))
                    potential_valid_blocks.append(Select(var, i) != model.eval(Select(var, i), model_completion=True))
            else:
                block.append(var != val)
                if not suffix and suffix not in ["ndim", "dtype"]: # potentially can add length too, TODO: asess
                    potential_valid_blocks.append(var != val)
        
        # Randomly block one of the constraints
        selected_const = block[rng.integers(len(block))]
        solver.add(selected_const)
        if print_details:
            print(f"Blocking constraint: {selected_const}")

        concrete_input, abstract_input = instantiate_args(model, definition["signature"], z3_args)
        status, exception_message = oracle_crash(driver, concrete_input, cpu=True)
 
        if status != "invalid":
            models.append(model)
            num_model += 1
            print(f"Valid models: {num_model}", end='\r', flush=True)
            # TODO: Check if this could be improved
            selected_valid_block = potential_valid_blocks[rng.integers(len(potential_valid_blocks))]
            solver.add(selected_valid_block)
            valid_blocks.append(selected_valid_block)
            if status != "nominal": # Always log crashes
                print(f"\n[{status}]\n{exception_message}")
                print(f"\nPotential bug. Input:\n{abstract_print(abstract_input, definition['signature'])}")
        elif print_details:
            print(abstract_print(abstract_input, definition["signature"]))
            print(f"\nThe input faced status {status}. Faced exception:\n{exception_message}")
        
        elapsed = time.time() - start

    return models
