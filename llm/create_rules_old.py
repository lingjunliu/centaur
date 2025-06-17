from utils.api_utils import get_signatures, get_driver
from utils.misc import map_torch_to_driver
from generator.input_generators import get_random_input
from eval.oracle import oracle_crash
from itertools import permutations
import os, re, time, glob, random, ast, types, inspect, difflib
import numpy as np
import google.generativeai as genai
import llm.valid_inputs as llm_inputs

def extract_python_code(response_text):
    match = re.search(r"```python(.*?)```", response_text, re.DOTALL)
    return match.group(1).strip() if match else None

def get_arity(z3_func):
    sig = inspect.signature(z3_func)
    return sum(1 for name in sig.parameters if name != "solver")

def define_z3_func(code_str):
    namespace = {}
    exec(code_str, namespace)
    funcs = [(k, v) for k, v in namespace.items() if isinstance(v, types.FunctionType) and k.endswith("_func") and "rule" in k]
    if not funcs:
        raise ValueError("No function matching criteria found.")
    return funcs[0][1]

def extract_lib_calls(func, lib):
    source = inspect.getsource(func)
    tree = ast.parse(source)
    calls = set()

    class CallVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute):
                names = []
                curr = node.func
                while isinstance(curr, ast.Attribute):
                    names.append(curr.attr)
                    curr = curr.value
                if isinstance(curr, ast.Name) and curr.id == lib:
                    names.append(curr.id)
                    full_name = ".".join(reversed(names))
                    calls.add(full_name)
            self.generic_visit(node)

    CallVisitor().visit(tree)
    return sorted(calls)

def check_rule(input_dict, arity, z3_func):
    if len(input_dict) < arity:
        return False
    for args in permutations(input_dict.keys(), arity):
        try:
            arg_dicts = tuple({k: input_dict[k]} for k in args)
            if z3_func(*arg_dicts):
                return True
        except:
            return False            
    return False

def deduplicate_similar_errors(error_msgs, threshold=0.7):
    unique_errors = []
    for msg in error_msgs:
        if not any(difflib.SequenceMatcher(None, msg, u).ratio() > threshold for u in unique_errors):
            unique_errors.append(msg)
    return unique_errors

def generate_rules(api, lib="torch", time_budget=30, seed=42):
    torch_to_driver, driver_to_torch = map_torch_to_driver()
    api_signature = get_signatures()[api]
    api_driver = get_driver(api, lib=lib)

    valid_inputs = []
    error_msg = []
    
    if driver_to_torch[api] in llm_inputs.generated_inputs:
        valid_inputs = llm_inputs.generated_inputs[driver_to_torch[api]]
   
    start_time = time.time()
    while time.time() - start_time < time_budget:
        rng = np.random.default_rng(seed)
        input_dict = get_random_input(api_signature, rng)
        status, exception_message = oracle_crash(api_driver, input_dict, cpu=True)

        if status == "invalid":
            if exception_message and exception_message not in error_msg:
                error_msg.append(exception_message)
        else:
            valid_inputs.append(input_dict)

        seed += 1
    
    if len(valid_inputs) == 0:
        return

    genai.configure(api_key=os.getenv("gemini_key"))
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
   
    error_msg = deduplicate_similar_errors(error_msg, threshold=0.7)
 
    for msg in error_msg:
        target_api = None
        apis = extract_lib_calls(api_driver, lib)

        prompt = f"What is the corresponding API for {api} among {apis}? Return one API name only."
        response = model.generate_content(prompt)
        target_api = response.text.strip().split()[0] 
            
        num = len(glob.glob("rules/rule_*.py")) + 1
        max_retries = 30 
        retry_count = 0

        print(f"Processing for {msg}")
        while retry_count < max_retries:
            prompt = f"The {target_api} API in {lib} library produces {msg}. "
            prompt += f"So, we define rule {num} to be a logical constraint that prevents this error.\n\n"

            prompt += "[Task Description]\n"
            prompt += f"Please generate Python code for this rule. It should implement 'rule_{num}_func' function. "
            prompt += "The function validates that inputs satisfy this rule (invariant learning phase), "
            prompt += "or adds the constraint to generate inputs that satisfy this rule (fuzz input generation phase).\n\n"

            prompt += "[Parameter Description]\n"
            prompt += f"The 'rule_{num}_func' function signature should be (arg1, .., arg{{N}}, solver=None). "
            prompt += "An argument is a dictionary {parameter_name: value}. "
            prompt += "For invariant learning phase, the 'value' is the actual value, and for fuzz input generation phase, "
            prompt += "it is a dictionary that contains different entries depending on the parameter type:\n"
            prompt += "  - [TYPE(s)]: [Dictionary Entries]\n"
            prompt += "  - integer, string: {'value': Int, 'dtype': Int}\n"
            prompt += "  - float: {'value': Real, 'dtype': Int}\n"
            prompt += "  - boolean: Bool (not a dictionary for booleans)\n"
            prompt += "  - tuple, list: {'length': Int, 'values': Array[Int, Int]}\n"
            prompt += "  - tensor, tensor_list: {'ndim': Int, 'shape': Array[Int, Int], 'dtype': Int, 'range': Array[Int, Int]}\n\n"

            prompt += "[Other Information]\n"
            prompt += "list_of_available_dtypes = [bool, np.int8, np.int16, np.int32, np.int64, np.uint8, np.float16, np.float32, np.float64, "
            prompt += "np.complex64, np.complex128, str, np.dtype]\n\n"

            ex_paths = random.sample(glob.glob("rule-examples/rule_*.py"), 5)
            ex_codes = [open(f).read() for f in ex_paths]

            prompt += "[Code Examples]\n"
            for i, code in enumerate(ex_codes, 1):
                prompt += f"Example {i}:\n{code}\n"

            prompt += "[Input Example]\n"
            prompt += f"{random.choice(valid_inputs)}\n\n"

            prompt += f"** REMINDER: The 'rule_{num}_func' function signature should be (arg1, .., arg{{N}}, solver=None). **" 

            response = model.generate_content(prompt)
            code_str = extract_python_code(response.text)
    
            if not code_str:
                print("No Python code found in the LLM response, retrying...")
                retry_count += 1
                continue
            try:
                z3_func = define_z3_func(code_str)
            except Exception as e:
                print(f"Error processing code: {e}, retrying...")
                retry_count += 1
                continue

            arity = get_arity(z3_func) 
            all_pass = all(check_rule(inp, arity, z3_func) for inp in valid_inputs)
            
            if all_pass:
                print("All valid inputs pass the generated rule!") 
                api_comment = f"# [{target_api}] {msg}"
                search_keyword = "Corresponds to rule"
                lines = code_str.splitlines()

                for i, line in enumerate(lines):
                    if search_keyword in line:
                        lines.insert(i, api_comment)
                        break
                annotated_code = "\n".join(lines)

                with open(f"rules/rule_{num}.py", "w") as f:
                    f.write(annotated_code)
                break
            else:
                print("Some valid inputs failed rule check, retrying...")
                retry_count += 1
