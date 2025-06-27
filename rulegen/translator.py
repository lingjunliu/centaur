import re
import os
from helpers import create_rule_expr, create_func_body

def create_py(header: str, directory: str = "../rules"):
    match = re.match(r"Rule\s+(\d+)\s+\((.*?)\)", header)
    if not match:
        raise ValueError(f"Could not extract rule number and description from header: {header}")

    rule_number = match.group(1)
    description = match.group(2)

    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f"rule_{rule_number}.py")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f'''import numpy as np

from utils.defaults import MAX_N_DIM, MAX_SZ_DIM, MAX_SZ_NUM, list_of_available_dtypes, list_of_string_values
from z3 import *

# {description} (Rule {rule_number})''')
    return rule_number

def create_func_template(rule_number, var_map, var_types):
    filename = f"../rules/rule_{rule_number}.py"

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
            elif t == "str":
                checks.append(f"isinstance({arg}, str)")
            else:
                raise ValueError(f"Unsupported type: {t}")
        return f"not ({' or '.join(checks)})"

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

rules = []
with open("rules", "r", encoding="utf-8") as f:
    content = f.read()

chunks = [chunk.strip() for chunk in content.split(">>") if chunk.strip()]
for chunk in chunks:
    match = re.match(r"(Rule\s+\d+\s+\(.*?\))\s*\n(.*)", chunk, re.DOTALL)
    if match:
        header = match.group(1).strip()
        rule_def = match.group(2).strip()
        rules.append((header, rule_def))

for i, (header, rule_def) in enumerate(rules, 1):
    rule_number = create_py(header)
    result = create_rule_expr(rule_number, rule_def)
    if result is None:
        continue
    var_map, var_types = result
    create_func_template(rule_number, var_map, var_types)
    create_func_body(rule_number, rule_def, var_map, var_types)
