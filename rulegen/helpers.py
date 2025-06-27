from lark import Lark
from UsedVarsCollector import UsedVarsCollector
from Z3ExprTransformer import Z3ExprTransformer
import re
import os

with open("grammar.lark", "r", encoding="utf-8") as f:
    grammar = f.read()

parser = Lark(grammar, start="start", parser="lalr", lexer="contextual")

def create_rule_expr(rule_number, rule_def):
    filename = f"../rules/rule_{rule_number}.py"
    try:
        bindings_text = re.findall(r"\{([^}]+)\}", rule_def)[0]
        bindings = [b.strip() for b in bindings_text.split(",")]
        var_types = {}
        var_names = []
        for i, b in enumerate(bindings):
            var, typ = [x.strip() for x in b.split(":")]
            var_types[var] = typ
            var_names.append(var)
        var_map = {v: f"arg{i+1}" for i, v in enumerate(var_names)}
        try:
            tree = parser.parse(rule_def)
        except Exception as e:
            if os.path.exists(filename):
                os.remove(filename)
            print(f"Parsing failed for rule {rule_number}\n{''.join(str(e).splitlines())}")
            return None
        try:
            transformer = Z3ExprTransformer(var_map, var_types)
            z3_expr = transformer.transform(tree)
        except Exception as e:
            if os.path.exists(filename):
                os.remove(filename)
            print(f"Transformation failed for rule {rule_number}\n{''.join(str(e).splitlines())}")
            return None

        code = f"\n\nrule_{rule_number} = lambda s, v, n=False: (\n    s.add(Not({z3_expr}) if n else\n          {z3_expr})\n)"
        with open(filename, "a", encoding="utf-8") as f:
            f.write(code)

    except Exception as e:
        if os.path.exists(filename):
            os.remove(filename)
        print(f"Unexpected error for rule {rule_number}: {e}")

    return var_map, var_types

def create_func_body(rule_number, rule_def, var_map, var_types):
    tree = parser.parse(rule_def)

    collector = UsedVarsCollector()
    collector.transform(tree)
    used_vars = collector.used_vars

    findent = " " * 4
    eindent = " " * 8

    lines = []
    lines.append(f"\n{eindent}# Variable declarations")
    lines.append(f"{eindent}solver = Solver()")

    for arg_name in var_map:
        arg = var_map[arg_name]
        typ = var_types.get(arg_name, "")
        entries = used_vars.get(arg_name, set())

        if "tensor" in typ:
            if "ndim" in entries:
                lines.append(f"{eindent}{arg}_ndim = Int('{arg}_ndim')")
            if "shape" in entries:
                lines.append(f"{eindent}{arg}_shape = Array('{arg}_shape', IntSort(), IntSort())")
            if "dtype" in entries:
                lines.append(f"{eindent}{arg}_dtype = Int('{arg}_dtype')")
            if "range" in entries:
                lines.append(f"{eindent}{arg}_range = Array('{arg}_range', IntSort(), IntSort())")
        elif typ == "int":
            if "value" in entries:
                lines.append(f"{eindent}{arg}_value = Int('{arg}_value')")
        elif typ == "float":
            if "value" in entries:
                lines.append(f"{eindent}{arg}_value = Real('{arg}_value')")
        elif typ == "bool":
            if "value" in entries:
                lines.append(f"{eindent}{arg}_value = Bool('{arg}_value')")
        elif typ == "str":
            if "value" in entries:
                lines.append(f"{eindent}{arg}_value = String('{arg}_value')") 

    lines.append(f"\n{eindent}# Value assignments")

    for arg_name in var_map:
        arg = var_map[arg_name]
        typ = var_types.get(arg_name, "")
        entries = used_vars.get(arg_name, set())

        if "tensor" in typ:
            if "ndim" in entries:
                lines.append(f"{eindent}solver.add({arg}_ndim == {arg}.ndim)")
            if "shape" in entries:
                lines.append(f"{eindent}for i in range({arg}.ndim):")
                lines.append(f"{eindent}{findent}{arg}_shape = Store({arg}_shape, i, {arg}.shape[i])")
            if "dtype" in entries:
                lines.append(f"{eindent}solver.add({arg}_dtype == list_of_available_dtypes.index({arg}.dtype))")
            if "range" in entries:
                lines.append(f"{eindent}{arg}_range = Store({arg}_range, 0, int(np.min({arg})))")
                lines.append(f"{eindent}{arg}_range = Store({arg}_range, 1, int(np.max({arg})))")
        elif typ == "int":
            if "value" in entries:
                lines.append(f"{eindent}solver.add({arg}_value == int({arg}))")
        elif typ in ["float", "bool"]:
            if "value" in entries:
                lines.append(f"{eindent}solver.add({arg}_value == {arg})")
        elif typ == "str":
            if "value" in entries:
                lines.append(f"{eindent}solver.add({arg}_value == list_of_string_values.index({arg}))")

    lines.append(f"\n{eindent}# Constraints for rule {rule_number}")
    dict_entries = []
    for arg_name in var_map:
        arg = var_map[arg_name]
        entries = used_vars.get(arg_name, set())
        for entry in entries:
            dict_entries.append(f"'{arg}_{entry}': {arg}_{entry}")

    dict_str = ", ".join(dict_entries)
    lines.append(f"{eindent}rule_{rule_number}(solver, {{{dict_str}}})")
    lines.append(f"{eindent}return solver.check() == sat\n")
    lines.append(f"{findent}# Fuzz input generation phase")
    lines.append(f"{findent}else:")

    dict_entries = []
    for arg_name in var_map:
        arg = var_map[arg_name]
        entries = used_vars.get(arg_name, set())
        for entry in entries:
            dict_entries.append(f"'{arg}_{entry}': {arg}['{entry}']")

    dict_str = ", ".join(dict_entries)
    lines.append(f"{eindent}rule_{rule_number}(solver, {{{dict_str}}}, neg)")

    func_body_str = "\n".join(lines)

    filename = f"../rules/rule_{rule_number}.py"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(func_body_str + "\n")
