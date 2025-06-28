from lark import Lark
import os, re, time, random, json
import google.generativeai as genai
from collections import defaultdict

with open("grammar.lark", "r", encoding="utf-8") as f:
    grammar = f.read()
parser = Lark(grammar)

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

def load_example_rules(path="examples", k=5):
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    examples = [(lines[i], lines[i + 1]) for i in range(0, len(lines), 2)]
    return random.sample(examples, min(k, len(examples)))

def log_response(label, prompt, response, num_failures=0):
    with open("log-rulegen", "a", encoding="utf-8") as log_file:
        log_file.write(">>> PROMPT\n")
        log_file.write(prompt.strip() + "\n\n")
        log_file.write("<<< RESPONSE\n")
        log_file.write(response.strip() + "\n")
        log_file.write(f"** {label.upper()} **")
        if num_failures:
            log_file.write(f" (num_failures: {num_failures})\n\n")
        else:
            log_file.write("\n\n")

def generate_rules(lib="torch", timeout=1800, max_failures=30, max_rules=30000):
    num_failures = 0
    num_rules = 1
    rule_defs = set()

    if os.path.exists("rules"):
        with open("rules", "r") as f:
            block = []
            for line in f:
                if line.strip() == ">>":
                    block = []
                else:
                    block.append(line.strip())
                    if len(block) == 2:
                        rule_defs.add(block[1])

    genai.configure(api_key=os.getenv("gemini_key"))
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    chat = model.start_chat(history=[])

    feedback = ""
    base_time = time.time()
    while time.time() - base_time < timeout and num_rules <= max_rules:
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
        prompt += f"Define Rule {num_rules} that some API parameters in {lib} should satisfy.\n\n"
        # prompt += f"Define Rule {num_rules} to suppress the following error message from {api} API in {lib}:\n"
        # prompt += f"\"{safe_error_msg}\"\n\n"

        prompt += "Type is encoded as an integer (index of the following list):\n"
        prompt += "[bool, np.int8, np.int16, np.int32, np.int64, np.uint8, np.float16, np.float32, np.float64, "
        prompt += "np.complex64, np.complex128, str, np.dtype]\n\n"
        prompt += "String value should be selected from the following list:\n"
        prompt += '["ii", "ii->i", "i,j->ij", "bij,bjk->bik", "...ij->...ji", "bn,anm,bm->ba", "none", '
        prompt += '"mean", "sum", "max", "constant", "tanh"]\n\n'

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

        new_rule_def = None
        new_rule = None
        redundant_vars = []

        for i in range(len(lines) - 1):
            if re.match(r"^Rule \d+ \(.+\)$", lines[i].strip()):
                header = re.sub(r'Rule\s+\d+', f'Rule {num_rules}', lines[i].strip())
                new_rule_def = lines[i + 1].strip()
                new_rule = f"{header}\n{new_rule_def}"

                bindings = re.search(r"\{([^}]+)\}", new_rule_def)
                if bindings: 
                    declared_vars = [v.strip().split(":")[0].strip() for v in bindings.group(1).split(",")]
                else:
                    declared_vars = []

                rule_expr = new_rule_def.split("|=")
                if len(rule_expr) > 1:
                    rule_expr = rule_expr[1].strip()
                else:
                    rule_expr = ""

                redundant_vars = [v for v in declared_vars if v not in rule_expr]
                break

        if new_rule_def is None:
            feedback = "The output format was incorrect. Please follow the output format strictly."
            num_failures += 1
            log_response("format error", prompt, response, num_failures)
            if num_failures >= max_failures:
                feedback = f"Failed {max_failures} times in a row. Try to generate a different rule." 
                num_failures = 0
            continue

        if redundant_vars:
            feedback = f"All variables should appear in the expression. Redundant variables: {', '.join(redundant_vars)}"
            num_failures += 1
            log_response("redundant variables", prompt, response, num_failures)
            if num_failures >= max_failures:
                feedback = f"Failed {max_failures} times in a row. Try to generate a different rule." 
                num_failures = 0
            continue

        if new_rule_def in rule_defs:
            feedback = "This rule already exists. Please generate a more diverse and novel rule."
            num_failures += 1
            log_response("duplicated rule", prompt, response, num_failures)
            if num_failures >= max_failures:
                feedback = f"Failed {max_failures} times in a row. Try to generate a different rule." 
                num_failures = 0
            continue

        try:
            parser.parse(new_rule_def)
        except Exception as e:
            feedback = f"The rule failed to parse with the grammar. Error: {str(e)}"
            num_failures += 1
            log_response("parsing error", prompt, response, num_failures)
            if num_failures >= max_failures:
                feedback = f"Failed {max_failures} times in a row. Try to generate a different rule." 
                num_failures = 0
            continue
    
        with open("rules", "a", encoding="utf-8") as f:
            f.write(">>\n" + new_rule + "\n")

        log_response("success", prompt, response)
        rule_defs.add(new_rule_def)
        num_rules += 1
        num_failures = 0
        # base_time = time.time()
        feedback = "The previous rule generation was successful. But, try to avoid rules that are too similar."

def main():
    generate_rules()

if __name__ == "__main__":
    main()
