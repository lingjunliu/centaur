import os
import sys
from lark import Transformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.defaults import MAX_N_DIM, list_of_string_values

'''
# For TensorFlow
list_of_string_values = [
    "ii", "ii->i", "i,j->ij", "bij,bjk->bik", "...ij->...ji", "bn,anm,bm->ba", "none",
    "mean", "sum", "max", "min", "prod", "relu", "tanh", "sigmoid", "softmax",
    "elu", "selu", "gelu", "swish", "constant", "linear", "softplus"
]
'''

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
    def dtype_type(self, _): return "dtype"
    def str_type(self, _): return "str"

    def tuple_type(self, items):
        inner_type = items[0]
        return f"tuple({inner_type})"
    
    def list_type(self, items):
        inner_type = items[0]
        return f"list({inner_type})"

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
        # if var != "i":
            # raise Exception(f" invalid variable {var}")
        return f"And([Implies({var} < ({end} + 1), {body}) for {var} in range({MAX_N_DIM})])"

    def exists(self, items):
        var, start, end, body = items
        # if var != "i":
            # raise Exception(f" invalid variable {var}")
        return f"Or([And({var} < ({end} + 1), {body}) for {var} in range({MAX_N_DIM})])"

    def if_expr(self, items):
        if len(items) > 2:
            cond, then_expr, else_expr = items
        else:
            cond, then_expr = items
            else_expr = "False"

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
        op_map = {"+": "+", "-": "-", "*": "*", "/": "/", "×": "*", "%": "%"}
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
        elif func_name == "dtype_":
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
        v = items[0].value.strip('"') 
        if v in list_of_string_values:
            return str(list_of_string_values.index(v))
        else:
            raise Exception(f" unsupported string '{v}'")

    def prim_var(self, items):
        v = str(items[0])
        typ = self.var_types.get(v, "")
        if not typ:
            return v
        elif "⊎" in typ:
            member_types = [t.strip() for t in typ.split("⊎")]
            allowed_types = {"int", "float", "bool", "str", "dtype"}
            if not all(t in allowed_types for t in member_types):
                raise Exception(f" Expected union of primitive types for '{v}', got '{typ}'")
        elif typ not in {"int", "float", "bool", "str", "dtype"}:
            raise Exception(f" Expected primitive type for '{v}', got '{typ}'")
        return f'v["{self.var_map[v]}_value"]'

    def tensor_var(self, items):
        v = str(items[0])
        typ = self.var_types.get(v, "")
        if typ != "tensor":
            raise Exception(f" Expected tensor type for '{v}', got '{typ}'")
        return self.var_map[v]

    def tuple_var(self, items):
        v = str(items[0])
        typ = self.var_types.get(v, "")
        if not typ.startswith("tuple") and not typ.startswith("list"):
            raise Exception(f" Expected tuple/list type for '{v}', got '{typ}'")
        return self.var_map[v]

    def tuple_access(self, items):
        var = str(items[0])
        access_expr = items[1]
        return access_expr.replace("{var}", var)

    def tuple_index(self, items):
        index_expr = items[0]
        return f'Select(v["{{var}}_values"], {index_expr})'

    def tuple_len(self, _):
        return f'v["{{var}}_length"]'
