import re
import os
from helpers import create_rule_expr, create_func_body
import sys

def create_py(header: str, directory: str = "../rules", lib="torch"):
    match = re.match(r"Rule\s+(\d+)\s+\((.*?)\)", header)
    if not match:
        raise ValueError(f"Could not extract rule number and description from header: {header}")

    rule_number = match.group(1)
    description = match.group(2)

    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f"rule_{rule_number}.py")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f'''import numpy as np
import torch 
import tensorflow as tf

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values_{lib}, np_dtype
from z3 import *

# {description} (Rule {rule_number})''')
    return rule_number, filename

def create_func_template(rule_number, var_map, var_types, filename):
    param_list = [var_map[var] for var in var_map]
    param_str = ", ".join(param_list + ["solver=None", "neg=False"])
    extract_lines = [f"    {arg} = next(iter({arg}.values()))" for arg in param_list]

    def get_type_check(arg, typ):
        checks = []
        types = [t.strip() for t in typ.split("⊎")]
        for t in types:
            if t == "tensor":
                checks.append(f"isinstance({arg}, np.ndarray)")
            elif t == "int":
                checks.append(f"(isinstance({arg}, (int, np.integer)) and not isinstance({arg}, bool))")
            elif t == "float":
                checks.append(f"isinstance({arg}, (float, np.floating))")
            elif t == "bool":
                checks.append(f"isinstance({arg}, bool)")
            elif t == "dtype":
                checks.append(f"(isinstance({arg}, torch.dtype) or isinstance({arg}, tf.dtypes.DType))")
            elif t == "str":
                checks.append(f"isinstance({arg}, str)")
            elif (t.startswith("tuple(") and t.endswith(")")) or (t.startswith("list(") and t.endswith(")")):
                inner = t[t.index("(")+1:-1]
                container_type = "tuple" if t.startswith("tuple(") else "list"
                if inner == "int":
                    inner_check = "(isinstance(e, (int, np.integer)) and not isinstance(e, bool))"
                elif inner == "float":
                    inner_check = "isinstance(e, (float, np.floating))"
                elif inner == "bool":
                    inner_check = "isinstance(e, bool)"
                elif inner == "str":
                    inner_check = "isinstance(e, str)"
                else:
                    raise ValueError(f"Unsupported {container_type} inner type: {inner}")
                checks.append(
                    f"(isinstance({arg}, {container_type}) and all({inner_check} for e in {arg}))"
                )
            else:
                raise ValueError(f"Unsupported type: {t}")
        if len(checks) > 1:
            return f"not ({' or '.join(checks)})"
        else:
            return f"not {checks[0]}"

    check_lines = []
    for var in var_map:
        arg = var_map[var]
        typ = var_types.get(var, "")
        if not typ:
            continue
        condition = get_type_check(arg, typ)
        check_lines.append(f"        if {condition}:")
        check_lines.append("            return False")

    func_code = f"""

def rule_{rule_number}_func({param_str}):
{chr(10).join(extract_lines)}

    # Invariant learning phase
    if not solver:
{chr(10).join(check_lines)}
"""
    with open(filename, "a", encoding="utf-8") as f:
        f.write(func_code)

def write_rules(dir, rules_file, lib="torch"):
    rules = []
    with open(rules_file, "r", encoding="utf-8") as f:
        content = f.read()

    chunks = [chunk.strip() for chunk in content.split(">>") if chunk.strip()]
    for chunk in chunks:
        match = re.match(r"(Rule\s+\d+\s+\(.*?\))\s*\n(.*)", chunk, re.DOTALL)
        if match:
            header = match.group(1).strip()
            rule_def = match.group(2).strip()
            rules.append((header, rule_def))

    for i, (header, rule_def) in enumerate(rules, 1):
        rule_number, rules_filename = create_py(header, directory=dir, lib=lib)
        result = create_rule_expr(rule_number, rule_def, filename=rules_filename, lib=lib)
        if result is None:
            continue
        var_map, var_types = result
        try:
            create_func_template(rule_number, var_map, var_types, rules_filename)
        except Exception as e:
            if os.path.exists(rules_filename):
                os.remove(rules_filename)
            print(f"Function template creation failed for rule {rule_number}\n{e}")
            continue
        try:
            create_func_body(rule_number, rule_def, var_map, var_types, rules_filename, lib=lib)
        except Exception as e:
            if os.path.exists(rules_filename):
                os.remove(rules_filename)
            print(f"Function body creation failed for rule {rule_number}\n{e}")
            continue

def main():
    lib = sys.argv[1] if len(sys.argv) > 1 else "torch"
    llm = sys.argv[2] if len(sys.argv) > 2 else "gemini"

    if llm == "gemini":
        base_dir = os.path.abspath(f"../rules-{lib}")
    elif llm == "openai":
        base_dir = os.path.abspath(f"{llm}/rules-{lib}")
    else:
        print("llm must be either 'gemini' or 'openai'")
        sys.exit(1)
    
    rules_file = "rules-ebnf"

    for entry in os.listdir(base_dir):
        sub_path = os.path.join(base_dir, entry)
        if os.path.isdir(sub_path):
            rule_path = os.path.join(sub_path, rules_file)
            if os.path.exists(rule_path):
                write_rules(sub_path, rule_path, lib=lib)

if __name__ == "__main__":
    main()
