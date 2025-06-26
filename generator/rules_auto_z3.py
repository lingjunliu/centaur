import numpy as np
import logging
import os
import re
import importlib.util
import inspect
from itertools import permutations

def get_rules_map(use_refernce=False):
    RULES_DIR = os.path.join(os.path.dirname(__file__), "..", "rules") if not use_refernce else os.path.join(os.path.dirname(__file__), "..", "references")
    rule_func_map = {}

    for filename in os.listdir(RULES_DIR):
        match = re.match(r"rule_(\d+)\.py", filename)
        if not match:
            continue

        rule_number = int(match.group(1))
        module_name = f"rule_{rule_number}"
        module_path = os.path.join(RULES_DIR, filename)

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"Failed to load {module_name}: {e}")
            continue

        func_name = f"{module_name}_func"
        if not hasattr(module, func_name):
            print(f"Function {func_name} not found in {module_name}")
            continue

        func = getattr(module, func_name)
        sig = inspect.signature(func)

        arity = sum(1 for p in sig.parameters if re.match(r"arg\d+", p))

        if arity not in rule_func_map:
            rule_func_map[arity] = {}
        rule_func_map[arity][module_name] = func
    
    return rule_func_map

def check_rules_z3(input_dict, print_rules=False, use_reference=False):
    rule_func_map = get_rules_map(use_reference=use_reference)
    set_of_rules_passed = set()
    
    if len(input_dict.keys()) < 1:
        print("Not enough arguments to check rules")
        return set_of_rules_passed
    
    for arity in rule_func_map:
        if len(input_dict.keys()) < arity:
            continue
        for args in permutations(input_dict.keys(), arity):
            for rule_name, z3_func in rule_func_map[arity].items():
                try:
                    arg_dicts = tuple({k: input_dict[k]} for k in args)
                    if z3_func(*arg_dicts):
                        if not any(
                            arity == a and rule_name == r and set(args) == set(arg_list)
                            for (a, r, *arg_list) in set_of_rules_passed
                        ):
                            set_of_rules_passed.add((arity, rule_name, *args))
                except:
                    pass 

    if print_rules:
        for arity, rule_name, *args in set_of_rules_passed:
            print(f"Arity {arity} Rule {rule_name} passed between {args}")
    return set_of_rules_passed
