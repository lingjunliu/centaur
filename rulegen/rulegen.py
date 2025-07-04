from lark import Lark
import os, re, time, random, json
import google.generativeai as genai
import torch, inspect, pkgutil, types, inspect
from collections import defaultdict

with open("grammar.lark", "r", encoding="utf-8") as f:
    grammar = f.read()
parser = Lark(grammar)

def list_all_apis(signature_path="../signatures.json"):
    def strip_suffix(api_name):
        parts = api_name.split(".")
        suffix_re = re.compile(r"^(.*?)(_\d+)?$")
        last_part = parts[-1]
        m = suffix_re.match(last_part)
        base_last = m.group(1) if m else last_part
        return ".".join(parts[:-1] + [base_last])

    with open(signature_path, "r") as f:
        signatures = json.load(f)

    apis = list({strip_suffix(k) for k in signatures.keys()})
    return apis 

def load_api_errors():
    err_file = os.path.join(os.path.dirname(__file__), "err_messages")
    api_to_errors = defaultdict(list)

    with open(err_file, "r") as f:
        content = f.read()

    blocks = [block.strip() for block in content.split(">>") if block.strip()]
    for block in blocks:
        lines = block.splitlines()
        if not lines:
            continue
        try:
            api, first_line = lines[0].split(", ", 1)
            error_msg = "\n".join([first_line] + lines[1:])
            api_to_errors[api].append(error_msg)
        except ValueError:
            continue

    return api_to_errors

api_list = list_all_apis()
api_to_errors = load_api_errors()

def get_doc_by_name(full_name):
    parts = full_name.split('.')
    try:
        obj = __import__(parts[0])
        for part in parts[1:]:
            obj = getattr(obj, part)
        return obj.__doc__ or ""
    except Exception:
        return ""

def load_example_rules(path="examples", k=5):
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    examples = [(lines[i], lines[i + 1]) for i in range(0, len(lines), 2)]
    return random.sample(examples, min(k, len(examples)))

def log_response(label, prompt, response, dir_path, num_failures=0):
    log_path = os.path.join(dir_path, "log-rulegen")
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(">>> PROMPT\n")
        log_file.write(prompt.strip() + "\n\n")
        log_file.write("<<< RESPONSE\n")
        log_file.write(response.strip() + "\n")
        log_file.write(f"** {label.upper()} **")
        if num_failures:
            log_file.write(f" (num_failures: {num_failures})\n\n")
        else:
            log_file.write("\n\n")

def generate_rules(api, max_time=30):
    num_failures = 0
    num_rules = 1
    rule_defs = set()

    dir_path = os.path.join("../rules", api)
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, "rules-ebnf")

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            block = []
            for line in f:
                line = line.strip()
                if line == ">>":
                    block = []
                else:
                    block.append(line)
                    if len(block) == 1:
                        match = re.match(r"Rule (\d+)", block[0])
                        if match:
                            num_rules = int(match.group(1)) + 1
                    if len(block) == 2:
                        rule_defs.add(block[1])

    genai.configure(api_key=os.getenv("gemini_key"))
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    chat = model.start_chat(history=[])

    feedback = ""
    base_time = time.time()
    while time.time() - base_time < max_time:
        prompt = ""
        if feedback:
            prompt += f"[Feedback Message from Prior Run]\n{feedback}\n\n"

        prompt += f"[Rule Grammar in EBNF Notation]"
        prompt += """
<rule> ::= "{" <binding_list> "}" "|=" <expr>

<binding_list> ::= <binding> ("," <binding>)*
<binding> ::= <VAR> ":" <type>

<type> ::= "tensor"
         | "int"
         | "float"
         | "bool"
         | "dtype"
         | "str"
         | "list" "(" <type> ")"
         | "tuple" "(" <type> ")"
         | <type> "⊎" <type>

<expr> ::= <and_expr>
<and_expr> ::= <or_expr> | <and_expr> "∧" <or_expr>
<or_expr> ::= <quant_expr> | <or_expr> "∨" <quant_expr>
<quant_expr> ::= "∀" <PRIMVAR> "∈" "[" <expr> "," <expr> "]" ":" <expr>
               | "∃" <PRIMVAR> "∈" "[" <expr> "," <expr> "]" ":" <expr>
               | <if_expr>
               | <compare_expr>

<if_expr> ::= "if" <expr> "then" <expr> [ "else" <expr> ]
<compare_expr> ::= <arith_expr> | <arith_expr> <COMPOP> <arith_expr>
<arith_expr> ::= <arith_expr> <ADDOP> <arith_term> | <arith_term>
<arith_term> ::= <arith_term> <MULOP> <arith_factor> | <arith_factor>
<arith_factor> ::= <TUPLEVAR> <tuple_access> | <func_call> | <constant> | <PRIMVAR> | "(" <expr> ")"

<tuple_access>  ::= "[" <expr> "]" | ".len"
<func_call> ::= <FUNC> "(" <TENSORVAR> [ "," <expr> ] ")"
<constant> ::= <NUMBER> | "true" | "false" | <STRING>
<COMPOP> ::= "=" | "≠" | ">" | "<" | "≥" | "≤"
<ADDOP> ::= "+" | "-"
<MULOP> ::= "*" | "/"
<FUNC> ::= "ndim" | "shape" | "dtype_" | "min" | "max"
<PRIMVAR> ::= any variable name (e.g., matches [a-zA-Z_][a-zA-Z_0-9]*)
<TENSORVAR> ::= same format as PRIMVAR
<TUPLEVAR> ::= same format as PRIMVAR

<VAR> ::= same format as PRIMVAR
<NUMBER> ::= any integer or decimal number (e.g., -5, 0.3, +7)
<STRING> ::= any quoted string (e.g., "hello", 'world')
"""

        # api = random.choice(list(api_to_errors.keys()))
        # error_msg = random.choice(api_to_errors[api])
        # safe_error_msg = error_msg.replace('"', '\\"')

        prompt += "\n[Task Description]\n"
        prompt += f"Define rules that {api} API parameters should satisfy. Refer to the API documentation. "
        
        errors = api_to_errors.get(api, [])
        if errors:
            prompt += "Particularly, there should be at least one rule to suppress each error message.\n\n"
        else:
            prompt += "\n\n"

        # prompt += f"Define Rule {num_rules} that API parameters in {lib} should satisfy.\n\n"
        # prompt += f"Define Rule {num_rules} to suppress the following error message from {api} API in {lib}:\n"
        # prompt += f"\"{safe_error_msg}\"\n\n"

        prompt += "Type is encoded as an integer (index of the following list):\n"
        prompt += "[bool, np.int8, np.int16, np.int32, np.int64, np.uint8, np.float16, np.float32, np.float64, "
        prompt += "np.complex64, np.complex128, str, np.dtype]\n\n"
        prompt += "String value should be selected from the following list:\n"
        prompt += '["ii", "ii->i", "i,j->ij", "bij,bjk->bik", "...ij->...ji", "bn,anm,bm->ba", "none", '
        prompt += '"mean", "sum", "max", "constant", "tanh"]\n\n'

        doc_str = get_doc_by_name(api)
        if doc_str:
            prompt += "[API Documentation]\n"
            prompt += doc_str.lstrip().rstrip() + "\n\n"

        if errors:
            prompt += "[Error Messages]\n"
            for err_msg in errors:
                prompt += err_msg + "\n"
            prompt += "\n"

        prompt += "[Output Format]\n"
        prompt += "Rule {Number} ({Description})\\n{Rule Definition}\n"
        prompt += "Ex) Rule 21 (primitive type variable should not be zero)\n"
        prompt += "    {v_1 : int ⊎ float} |= v_1 ≠ 0\n\n"

        prompt += "[Example Rules]\n"
        for desc, rule in load_example_rules():
            prompt += f"{desc}\n{rule}\n\n"

        prompt += "** IMPORTANT: The rule definition should be a new one and strictly follow the grammar. **\n"
        prompt += "** IMPORTANT: Rules should span diverse types (tensor, int, float, bool, dtype, str, tuple, list, union), properties, and numbers of parameters. **\n"
        # prompt += f"** IMPORTANT: Bindings should be from {{{params_str}}} and include only variables that are used in the expression. **\n"
        prompt += "** IMPORTANT: Variables should be named v_1, v_2, and so on. **\n"
        prompt += "** IMPORTANT: Bindings should be API parameters and include only variables that are used in the expression. **\n"

        response = chat.send_message(prompt)
        response = response.text.strip().replace('\u2212', '-').replace(' else true', '')
        lines = response.splitlines()

        feedback_messages = []
        new_rules = []
        pattern = re.compile(r"^Rule \d+ \(.+\)$")

        i = 0
        while i < len(lines) - 1:
            line = lines[i].strip()
            if pattern.match(line):
                header = re.sub(r'Rule\s+\d+', f'Rule {num_rules}', line)
                rule_def = lines[i + 1].strip()
                rule = f"{header}\n{rule_def}"
                new_rules.append((rule_def, rule))
                num_rules += 1
                i += 2
            else:
                i += 1

        if not new_rules:
            num_failures += 1
            msg = "No valid rules detected. Please follow the expected output format for each rule."
            feedback_messages.append(msg)
            log_response("format error", prompt, response, dir_path, num_failures)
            feedback = "\n".join(feedback_messages)
            continue

        for rule_def, rule_text in new_rules:
            bindings = re.search(r"\{([^}]+)\}", rule_def)
            if bindings:
                declared_vars = [v.strip().split(":")[0].strip() for v in bindings.group(1).split(",")]
            else:
                declared_vars = []

            rule_expr = rule_def.split("|=")
            rule_expr = rule_expr[1].strip() if len(rule_expr) > 1 else ""

            redundant_vars = [v for v in declared_vars if v not in rule_expr]
            if redundant_vars:
                msg = f"Redundant variables: {rule_def} (Unused: {', '.join(redundant_vars)})"
                num_failures += 1
                feedback_messages.append(msg)
                log_response("redundant variables", prompt, rule_text, dir_path, num_failures)
                continue

            if rule_def in rule_defs:
                msg = f"Duplicated rule: {rule_def}"
                num_failures += 1
                feedback_messages.append(msg)
                log_response("duplicated rule", prompt, rule_text, dir_path, num_failures)
                continue

            try:
                parser.parse(rule_def)
            except Exception as e:
                msg = f"Parse error: {rule_def} (Error: {str(e)})"
                num_failures += 1
                feedback_messages.append(msg)
                log_response("parsing error", prompt, rule_text, dir_path, num_failures)
                continue

            with open(file_path, "a", encoding="utf-8") as f:
                f.write(">>\n" + rule_text + "\n")

            log_response("success", prompt, rule_text, dir_path)
            rule_defs.add(rule_def)

        feedback = "\n".join(feedback_messages)

def main():
    for api in api_list:
        generate_rules(api)

if __name__ == "__main__":
    main()
