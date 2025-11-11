from lark import Transformer

class UsedVarsCollector(Transformer):
    def __init__(self):
        self.used_vars = {}

    def prim_var(self, items):
        var = str(items[0])
        self.used_vars.setdefault(var, set()).add("value")
        return var

    def tensor_var(self, items):
        var = str(items[0])
        return var

    def tuple_var(self, items):
        var = str(items[0])
        return var

    def tuple_access(self, items):
        var = str(items[0])
        access = str(items[1])
        if access == "len":
            self.used_vars.setdefault(var, set()).add("length")
        else:
            self.used_vars.setdefault(var, set()).add("values")
        return var  

    def tuple_index(self, items):
        return "index"

    def tuple_len(self, _):
        return "len"

    def func_call(self, items):
        func_name = str(items[0])
        var = str(items[1])

        if func_name == "min" or func_name == "max":
            prop = "range"
        else:
            prop = func_name
        self.used_vars.setdefault(var, set()).add(prop)
        return (func_name, var)
