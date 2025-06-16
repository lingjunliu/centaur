from google import genai
import os, subprocess, time
import numpy as np
from utils.new_api_utils import get_n_variations, get_signature
from utils.misc import read_file_in_root, bcolors
from llm.create_driver import fetch_documentation, extract_code_from_response, extract_function_info
from llm.valid_inputs import generated_inputs

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

def get_prompt(torch_api, lib="torch", suffix=0):
    doc = extract_function_info(fetch_documentation(torch_api), torch_api)
    signature = get_signature(torch_api, lib=lib, suffix=suffix)
    prefix = f'This is the documentation for the function {torch_api}:\n\n"{doc.encode('ascii', errors='ignore').decode()}"\n\n' if doc else ""
    key = torch_api if suffix == 0 else f"{torch_api}_{suffix}"
    with open(f"{CUR_DIR}/prompt_input_gen.md", "r", encoding="utf-8") as file:
        prompt = file.read()
        prompt = prompt.replace("{api}", torch_api)
        prompt = prompt.replace("{key}", key)
        prompt = prompt.replace("{signature}", str(signature))
    return prefix + prompt

def save_and_run_code(torch_api, code, suffix=0, lib="torch"):
    key = torch_api if suffix == 0 else f"{torch_api}_{suffix}"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    validity_checker_code = f"""
from utils.new_api_utils import run_api

generated_inputs = dict()

{code}

def check_valid(api, list_of_inputs, lib="torch"):
    for idx, input_dict in enumerate(list_of_inputs):
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    print("Valid")

if '{key}' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key '{key}'.")

check_valid('{torch_api}', generated_inputs['{key}'], lib="{lib}")
"""
    
    filepath = f"{CUR_DIR}/inputs/{torch_api.split('.')[-1]}_{suffix}.py"
    with open(filepath, 'w') as f:
        f.write(validity_checker_code)
    
    try:
        # Run the generated file with a timeout of 30 sec just in case
        result = subprocess.run(['python', '-m', f'llm.inputs.{torch_api.split('.')[-1]}_{suffix}'], capture_output=True, text=True, timeout=30)
        
        return result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        print(f"Execution of {torch_api} timed out.")
        return "", "Timeout: Execution could not be completed in 30 seconds."

def retry_prompt(error):
    prompt = f"""Error faced during execution: {error}.
Please fix the error and retry the input generation. Only provide the code, skip any other text. Do not include verbose comments inside code. If you feel like you have added too many inputs and some of them are causing validity errors, remove them. If you do not feel confident about the error, try to generate a new input and delete the old one.
    """
    return prompt

def generate_inputs(api, suffix=0, max_attempts=5, lib="torch"):
    model = "gemini-2.0-flash"
    gemini_key = os.getenv("gemini_key")

    print("Running code generation after 6 seconds...")
    time.sleep(6)
    client = genai.Client(api_key=gemini_key)
    chat = client.chats.create(model=model)
    response = chat.send_message(get_prompt(api, lib=lib, suffix=suffix))
    print("Got response from Gemini API.")
    code = extract_code_from_response(response.text)    
    output, error = save_and_run_code(api, code, suffix=suffix, lib=lib)
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
        output, error = save_and_run_code(api, code, suffix=suffix, lib=lib)
        attempt += 1
        
        if attempt >= max_attempts:
            print("Max attempts reached. Exiting.")
            break
        
    if output.endswith("Valid"):
        print("\nInput generated successfully.")
        code = code.replace("generated_inputs = {}", "")
        with open(f"{CUR_DIR}/valid_inputs.py", "a") as fv:
            fv.write(code + "\n\n")
    else:
        print(f"\nInput generation failed after {max_attempts} attempts.")
        
    return [api, get_torch_api(api)] + to_return

def main():
    lib = "torch"
    torch_apis = read_file_in_root("torch_apis.txt")
    total = len(torch_apis)
    durations = []
    
    for idx, torch_api in enumerate(torch_apis):
        n_variations = get_n_variations(torch_api, lib=lib)
        start = time.time()
        generated = True
        if n_variations > 1:
            for i in range(1, n_variations+1):
                key = f"{torch_api}_{i}"
                if key in generated_inputs:
                    generated = False
                    continue
                print(f"\nGenerating valid inputs for {torch_api}_{i}...\n")
                result = generate_inputs(torch_api, suffix=i, lib=lib)
                with open(f"{CUR_DIR}/inputs.csv", "a") as f:
                    f.write(",".join(map(str, result)) + "\n")            
        else:
            if torch_api in generated_inputs:
                generated = False
                continue
            print(f"\nGenerating valid inputs for {torch_api}...\n")
            result = generate_inputs(torch_api, lib=lib)
            with open(f"{CUR_DIR}/inputs.csv", "a") as f:
                f.write(",".join(map(str, result)) + "\n")
        
        if generated:
            durations.append(time.time()-start)
        print(f"{bcolors.OKGREEN}Done with {idx+1}/{total} | ETR: {(total-idx-1)*np.mean(durations):.2f}s{bcolors.ENDC}")
        
if __name__ == "__main__":
    main()