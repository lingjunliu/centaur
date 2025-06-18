from lark import Lark
import os, re, time, random
import google.generativeai as genai

with open("grammar.lark", "r", encoding="utf-8") as f:
    grammar = f.read()
parser = Lark(grammar)

def load_example_rules(path="examples", k=5):
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    examples = [(lines[i], lines[i + 1]) for i in range(0, len(lines), 2)]
    return random.sample(examples, min(k, len(examples)))

def log_response(label, prompt, response):
    with open("log-rulegen", "a", encoding="utf-8") as log_file:
        log_file.write(f"[{label.upper()}]\n")
        log_file.write(">>> PROMPT\n")
        log_file.write(prompt.strip() + "\n\n")
        log_file.write("<<< RESPONSE\n")
        log_file.write(response.strip() + "\n\n")

def generate_rules(lib="torch", timeout=300, max_failures=30, max_rules=300):
    num_failures = 0
    num_rules = 0
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
    while time.time() - base_time < timeout and num_failures < max_failures and num_rules < max_rules:
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
         | "str"
         | <type> "⊎" <type>

<expr> ::= <and_expr>
<and_expr> ::= <or_expr> | <and_expr> "∧" <or_expr>
<or_expr> ::= <quant_expr> | <or_expr> "∨" <quant_expr>
<quant_expr> ::= "∀" <VAR> "∈" "[" <expr> "," <expr> "]" ":" <expr>
               | "∃" <VAR> "∈" "[" <expr> "," <expr> "]" ":" <expr>
               | <if_expr>
               | <compare_expr>

<if_expr> ::= "if" <expr> "then" <expr> "else" <expr>
<compare_expr> ::= <arith_expr> | <arith_expr> <COMPOP> <arith_expr>
<arith_expr> ::= <arith_expr> <ADDOP> <arith_term> | <arith_term>
<arith_term> ::= <arith_term> <MULOP> <arith_factor> | <arith_factor>
<arith_factor> ::= <func_call> | <constant> | <PRIMVAR> | "(" <expr> ")"

<func_call> ::= <FUNC> "(" <TENSORVAR> [ "," <expr> ] ")"
<constant> ::= <NUMBER> | "true" | "false" | <STRING>
<COMPOP> ::= "=" | "≠" | ">" | "<" | "≥" | "≤"
<ADDOP> ::= "+" | "-"
<MULOP> ::= "*" | "/"
<FUNC> ::= "ndim" | "shape" | "dtype" | "min" | "max"
<PRIMVAR> ::= any variable name (e.g., matches [a-zA-Z_][a-zA-Z_0-9]*)
<TENSORVAR> ::= same format as PRIMVAR
<VAR> ::= same format as PRIMVAR
<NUMBER> ::= any integer or decimal number (e.g., -5, 0.3, +7)
<STRING> ::= any quoted string (e.g., "hello", 'world')
"""

        prompt += "\n[Task Description]\n"
        prompt += f"Define Rule {num_rules} that some API parameters in {lib} should satisfy.\n\n"

        prompt += "Type is encoded as an integer (index of the following list):\n"
        prompt += "[bool, np.int8, np.int16, np.int32, np.int64, np.uint8, np.float16, np.float32, np.float64, "
        prompt += "np.complex64, np.complex128, str, np.dtype]\n\n"

        prompt += "[Output Format]\n"
        prompt += "Rule {Number} ({Description})\\n{Rule Definition}\n"
        prompt += "Ex) Rule 21 (primitive type variable should not be zero)\n"
        prompt += "    {v_1 : int ⊎ float} |= v_1 ≠ 0\n\n"

        prompt += "[Example Rules]\n"
        for desc, rule in load_example_rules():
            prompt += f"{desc}\n{rule}\n\n"

        prompt += "** IMPORTANT: The rule definition should be a new one and strictly follow the grammar. **\n"
        prompt += "** IMPORTANT: Prioritize rules on tensors, but still try to diversify types (int, float, bool, str, and union). **\n"
        prompt += "** IMPORTANT: Variables should be named v_1, v_2, and so on. **\n"
        prompt += "** IMPORTANT: Bindings should include only variables that are used in the expression. **\n"

        response = chat.send_message(prompt)
        response = response.text.strip().replace('\u2212', '-')
        lines = response.splitlines()

        new_rule_def = None
        new_rule = None

        for i in range(len(lines) - 1):
            if re.match(r"^Rule \d+ \(.+\)$", lines[i].strip()):
                header = lines[i].strip()
                new_rule_def = lines[i + 1].strip()
                new_rule = f"{header}\n{new_rule_def}"
                break

        if new_rule_def is None:
            feedback = "The output format was incorrect. Please follow the output format strictly."
            log_response("format error", prompt, response)
            num_failures += 1
            continue

        if new_rule_def in rule_defs:
            feedback = "This rule already exists. Please generate a more diverse and novel rule."
            log_response("duplicated rule", prompt, response)
            num_failures += 1
            continue

        try:
            parser.parse(new_rule_def)
        except Exception as e:
            feedback = f"The rule failed to parse with the grammar. Error: {str(e)}"
            log_response("parsing error", prompt, response)
            num_failures += 1
            continue

        with open("rules", "a", encoding="utf-8") as f:
            f.write(">>\n" + new_rule + "\n")

        log_response("success", prompt, response)
        rule_defs.add(new_rule_def)
        num_rules += 1
        num_failures = 0
        base_time = time.time()
        feedback = "The previous rule generation was successful."

def main():
    generate_rules()

if __name__ == "__main__":
    main()
