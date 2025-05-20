from google import genai
import os, subprocess, time
from utils.api_utils import get_signatures
from llm.create_driver import fetch_documentation, extract_code_from_response, extract_function_info

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def get_torch_api(api):
    torch_api = None
    with open(f"{CUR_DIR}/supported.csv", "r") as f:
        for line in f.readlines():
            tokens = line.strip().split(",")
            if tokens[0] == api:
                torch_api = tokens[1]
                break
    if torch_api is None:
        with open(f"{CUR_DIR}/drivers_to_api.csv", "r") as f:
            for line in f.readlines():
                driver, cur_api = line.strip().split(',')
                if driver == api:
                    torch_api = cur_api
                    break
    return torch_api

def get_prompt(api):
    torch_api = get_torch_api(api)    
    if not torch_api:
        return None
    doc = extract_function_info(fetch_documentation(torch_api), torch_api)
    signature = get_signatures()[api]
    prefix = f'This is the documentation for the function {torch_api}:\n\n"{doc.encode('ascii', errors='ignore').decode()}"\n\n' if doc else ""
    with open(f"{CUR_DIR}/prompt_input_gen.md", "r", encoding="utf-8") as file:
        prompt = file.read()
        prompt = prompt.replace("{api}", torch_api)
        prompt = prompt.replace("{signature}", str(signature))
    return prefix + prompt

def save_and_run_code(api, code):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    validity_checker_code = f"""
from utils.api_utils import get_driver
from eval.oracle import oracle_crash

{code}

from utils.api_utils import get_driver
from eval.oracle import oracle_crash

def check_valid(api, list_of_inputs, lib="torch"):
    api_driver = get_driver(api, lib=lib)
    for idx, input_dict in enumerate(list_of_inputs):
        api_driver(input_dict, cpu=True)
    
    print("Valid")

check_valid('{api}', generated_inputs)
"""
    
    filepath = f"{CUR_DIR}/inputs/{api}.py"
    with open(filepath, 'w') as f:
        f.write(validity_checker_code)
    
    try:
        # Run the generated file with a timeout of 30 sec just in case
        result = subprocess.run(['python', '-m', f'llm.inputs.{api}'], capture_output=True, text=True, timeout=30)
        
        return result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        print(f"Execution of {api} timed out.")
        return "", "Timeout: Execution could not be completed in 30 seconds."

def retry_prompt(error):
    prompt = f"""Error faced during execution: {error}.
Please fix the error and retry the input generation. Only provide the code, skip any other text. Do not include verbose comments inside code. If you feel like you have added too many inputs and some of them are causing validity errors, remove them. If you do not feel confident about the error, try to generate a new input and delete the old one.
    """
    return prompt

def generate_inputs(api, max_attempts=5):
    model = "gemini-2.0-flash"
    gemini_key = os.getenv("gemini_key")

    print("Running code generation after 6 seconds...")
    time.sleep(6)
    client = genai.Client(api_key=gemini_key)
    chat = client.chats.create(model=model)
    response = chat.send_message(get_prompt(api))
    print("Got response from Gemini API.")
    code = extract_code_from_response(response.text)    
    output, error = save_and_run_code(api, code)
    attempt = 0
    to_return = [0] * max_attempts
    
    while not output.endswith("Valid"):
        print(f"Attempt {attempt + 1}: \n{error}")
        to_return[attempt] = 1
        print("Retrying code generation after 6 seconds...")
        time.sleep(6)
        response = chat.send_message(retry_prompt(error))
        print("Got response from Gemini API.")
        code = extract_code_from_response(response.text)
        output, error = save_and_run_code(api, code)
        attempt += 1
        
        if attempt >= max_attempts:
            print("Max attempts reached. Exiting.")
            break
        
    if output.endswith("Valid"):
        print("\nInput generated successfully.")
        with open(f"{CUR_DIR}/valid_inputs.py", "a") as fv:
            fv.write(code + "\n\n")
    else:
        print("\nInput generation failed after multiple attempts.")
        
    return [api, get_torch_api(api)] + to_return

def main():
    with open(f"{CUR_DIR}/needs_inputs.txt", "r") as f:
        apis = [line.strip() for line in f.readlines()]
    
    for api in apis:
        print(f"\nGenerating valid inputs for {api}...\n")
        result = generate_inputs(api)
        with open(f"{CUR_DIR}/inputs.csv", "a") as f:
            f.write(",".join(map(str, result)) + "\n")
        
if __name__ == "__main__":
    main()