from z3 import *
from functools import reduce
import numpy as np
from .defaults import *
from generator.input_generators import get_ll
from generator.rules_auto_z3 import get_rules_map

def add_negative_buckets(buckets):
    for element in buckets:
        if element > 0:
            buckets.append(-1*element)
    return sorted(buckets)

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
            z3_args[param] = {
                "value": Bool(f"{param}_value")
            }
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
            z3_args[param] = {
                "value": Int(f"{param}_value")
            }
        else:
            raise ValueError(f"Unsupported type: {typ}")
    return z3_args

def parition_solvers(solver, signature, z3_args, lib="torch", rng=np.random.default_rng(42)):
    """
    This function partitions the solver into two solvers per boolean argument.
    One solver will assert the boolean argument to be True, and the other will assert it to be False.
    For each solver, the range of the tensor values is chosen.
    """
    solvers = []
    for param_name, z3_var in z3_args.items():
        if signature[param_name] == "boolean":
            value = z3_var['value']
            solver_true.add(*solver.assertions())
            solver_true.add(value == True)
            solver_false.add(*solver.assertions())
            solver_false.add(value == False)
            solvers.append(solver_true)
            solvers.append(solver_false)
    
    if len(solvers) == 0:
        solvers.append(solver)
    
    for solver in solvers:
        for param_name, z3_var in z3_args.items():
            if signature[param_name] == "tensor" or signature[param_name] == "tensor_list":
                range_ = z3_var['range']
                # Choosing low and high values for the tensor range
                buckets = sorted(rng.choice(add_negative_buckets(int_buckets), size=4, replace=False))
                low, high = rng.randint(buckets[0], buckets[1]), rng.randint(buckets[2], buckets[3])
                solver.add(Select(range_, 0) == low)
                solver.add(Select(range_, 1) == high)
    
    return solvers

def initial_constraints(solver, signature, z3_args, lib="torch"):
    domain_limits = domain_limits_torch if lib == "torch" else domain_limits_tf
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
            value = z3_var['value']
            solver.add(Or(value == domain_limits[f'{param_type}_value_range'][0], value == domain_limits[f'{param_type}_value_range'][1]))            # Two possible values, True or False
        elif param_type == "dtype":
            value = z3_var['value']
            solver.add(And(value >= 0, value <= len(list_of_available_dtypes) - 3)) 

def collect_constraints(solver, api, ruleset, z3_args, use_reference=False, lib="torch"):
    rule_func_map = get_rules_map(api, use_reference=use_reference, lib=lib)
    for rule in ruleset:
        arity, rule_name, *args = rule
        rule_func = rule_func_map[arity][rule_name]
        
        arg_dicts = []
        for param_name in args:
            arg_dicts.append({param_name: z3_args[param_name]})
       
        rule_func(*arg_dicts, solver=solver)

def collect_neg_constraint(solver, api, rule, z3_args, use_reference=False):
    rule_func_map = get_rules_map(api, use_reference=use_reference)
    arity, rule_name, *args = rule
    rule_func = rule_func_map[arity][rule_name]
        
    arg_dicts = []
    for param_name in args:
        arg_dicts.append({param_name: z3_args[param_name]})
       
    rule_func(*arg_dicts, solver=solver, neg=True)

def instantiate_args(model, signature, z3_args, seed=42, lib="torch"):
    concrete_args = {}
    abstract_args = {}
    rng = np.random.default_rng(seed)

    list_of_string_values = list_of_string_values_torch if lib == "torch" else list_of_string_values_tf

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
            value = model.eval(z3_var['value'], model_completion=True).as_long()
            concrete_args[param_name] = list_of_available_dtypes[value]
        elif param_type == "boolean":
            value = is_true(model.eval(z3_var['value'], model_completion=True))
            concrete_args[param_name] = is_true(value)

        abstract_args[param_name] = get_ll(param_type, concrete_args[param_name])

    return concrete_args, abstract_args