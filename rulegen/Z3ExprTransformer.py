import os
import sys
from lark import Transformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.defaults import MAX_N_DIM

class Z3ExprTransformer(Transformer):
    def __init__(self, var_map, var_types):
        self.var_map = var_map
        self.var_types = var_types

    def start(self, items):
        return items[0]

    def rule(self, items):
        return items[1]

    def binding_list(self, items):
        return items

    def binding(self, items):
        var_name = str(items[0])
        var_type = items[1]
        return (var_name, var_type)

    def tensor_type(self, _): return "tensor"
    def int_type(self, _): return "int"
    def float_type(self, _): return "float"
    def bool_type(self, _): return "bool"
    def str_type(self, _): return "str"

    def union_type(self, items):
        left, right = items
        return f"({left} | {right})"

    def expr(self, items):
        return items[0]

    def and_expr_base(self, items):
        return items[0]

    def and_(self, items):
        return f"And({items[0]}, {items[1]})"

    def or_expr_base(self, items):
        return items[0]

    def or_(self, items):
        return f"Or({items[0]}, {items[1]})"

    def forall(self, items):
        var, start, end, body = items
        var_name = str(var)
        if var_name != "i":
            raise Exception(f" invalid variable {var_name}")
        return f"And([Implies(i < ({end} + 1), {body}) for i in range({MAX_N_DIM})])"

    def exists(self, items):
        var, start, end, body = items
        var_name = str(var)
        if var_name != "i":
            raise Exception(f" invalid variable {var_name}")
        return f"Or([And(i < ({end} + 1), {body}) for i in range({MAX_N_DIM})])"

    def if_expr(self, items):
        cond, then_expr, else_expr = items
        return f"If({cond}, {then_expr}, {else_expr})"

    def compare_expr_base(self, items):
        return items[0]

    def compare(self, items):
        left, op, right = items
        op_map = {
            "=": "==", "≠": "!=", "!=": "!=", "<": "<", ">": ">", "≤": "<=", "<=": "<=", "≥": ">=", ">=": ">="
        }
        op_str = op_map[str(op)]
        return f"{left} {op_str} {right}"
    
    def binop(self, items):
        left, op, right = items
        op_map = {"+": "+", "-": "-", "*": "*", "/": "/", "×": "*"}
        op_str = op_map[str(op)]
        return f"{left} {op_str} {right}"

    def arith_expr_base(self, items):
        return items[0]

    def arith_term_base(self, items):
        return items[0]

    def arith_func_call(self, items):
        return items[0]

    def arith_constant(self, items):
        return items[0] 

    def arith_var(self, items):
        return items[0] 
    
    def parens(self, items):
        return f"({items[0]})"

    def func_call(self, items):
        func_name = str(items[0])
        var = items[1]
        index_expr = items[2] if len(items) > 2 else "0"

        if func_name == "ndim":
            return f'v["{var}_ndim"]'
        elif func_name == "dtype":
            return f'v["{var}_dtype"]'
        elif func_name == "shape":
            return f'Select(v["{var}_shape"], {index_expr})'
        elif func_name == "min":
            return f'Select(v["{var}_range"], 0)'
        elif func_name == "max":
            return f'Select(v["{var}_range"], 1)'
        else:
            raise Exception(f" {func_name} not supported")

    def number(self, items):
        return str(items[0])

    def true(self, _):
        return "True"

    def false(self, _):
        return "False"

    def string(self, items):
        return str(items[0])

    def prim_var(self, items):
        v = str(items[0])
        if v == "i":
            return v
        typ = self.var_types.get(v, "")
        if "tensor" in typ:
            raise Exception(f" tensor type variable {v}")
        else:
            return f'v["{self.var_map[v]}_value"]'

    def tensor_var(self, items):
        v = str(items[0])
        return self.var_map[v]
