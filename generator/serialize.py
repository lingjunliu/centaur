from z3 import *
import json

def extract_z3_variables(z3_args):
    variables = {}
    def recursive_extract(item):
        if isinstance(item, dict):
            for key, value in item.items():
                recursive_extract(value)
        else:
            variables[item.decl().name()] = item
    for key, value in z3_args.items():
        recursive_extract(value) 
    return variables

def load_model(model_data, z3_args):
    solver = Solver()
    variables = extract_z3_variables(z3_args)
    for name, info in model_data.items():
        if name not in variables:
            print(f"Warning: Variable {name} not found in z3_args.")
            continue
        var = variables[name]
        if info["type"] == "Int":
            solver.add(var == int(info["value"]))
        elif info["type"] == "Real":
            solver.add(var == RealVal(info["value"]))
        elif info["type"] == "Bool":
            solver.add(var == BoolVal(info["value"]))
        elif info["type"] == "Array":
            val = info["value"]
            parsed_val = parse_smt2_string(f"(declare-fun a () (Array Int Int)) (assert (= a {val}))")
            solver.add(var == parsed_val[0].children()[1])
        else:
            raise ValueError(f"Unknown type {info['type']} for {name}")
    
    if solver.check() == sat:
        return solver.model()
    else:
        return None

def serialize_z3_value(val):
    if isinstance(val, z3.IntNumRef):
        return {"type": "Int", "value": val.as_long()}
    elif isinstance(val, z3.RatNumRef):
        return {"type": "Real", "value": str(val)}
    elif isinstance(val, z3.BoolRef):
        return {"type": "Bool", "value": is_true(val)}
    elif isinstance(val, z3.ArrayRef):
        return {"type": "Array", "value": val.sexpr()}
    else:
        return {"type": "Unknown", "value": str(val)}

def save_model(model, path):
    serialized = {}
    for d in model.decls():
        name = d.name()
        val = model[d]
        serialized[name] = serialize_z3_value(val)
    with open(path, "w") as f:
        json.dump(serialized, f, indent=2)
