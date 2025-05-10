import time
import numpy as np
from z3 import *
from .input_generators import get_ll
from .rules_z3 import rule_func_map
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
           
            np_array = np.random.uniform(low, high, size=shape).astype(list_of_available_dtypes[dtype])
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

    while elapsed < model_gen_duration and num_model < max_model:
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

        if status == "nominal":
            models.append(model)
            num_model += 1
            print(f"Valid models: {num_model}", end='\r', flush=True)
        
        elapsed = time.time() - start

    return models
