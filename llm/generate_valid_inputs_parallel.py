from google import genai
import os, subprocess, time
from multiprocessing import Pool

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
from utils.new_api_utils import get_n_variations, get_signature, get_doc_tf, get_api_suffix
from utils.misc import read_file_in_root, bcolors
import llm.valid_inputs_torch as valid_inputs_torch
import llm.valid_inputs_tf as valid_inputs_tf
from llm.create_driver import fetch_documentation, extract_code_from_response, extract_function_info
import logging
import sys

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

def get_prompt(api, lib="torch", suffix=0):
    examples = {
        "torch": """
```python
import torch, copy

def addcmul_inputs():
    list_of_inputs = []
    # Input 1, valid
    input = torch.tensor([1.0, 2.0, 3.0]).numpy()   # tensor
    tensor1 = torch.tensor([0.1, 0.2, 0.3]).numpy() # tensor
    tensor2 = torch.tensor([10.0, 20.0, 30.0]).numpy()  # tensor
    value = 2.0 # float
    out = torch.tensor().numpy()    # tensor

    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value,
        "out": out
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))
    
    # Input 2, valid
    input = torch.ones((2, 3)).numpy()
    tensor1 = torch.tensor([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]]).numpy()
    tensor2 = torch.tensor([[0.1, 0.2, 0.3],
                            [0.4, 0.5, 0.6]]).numpy()
    value = 0.5
    out = torch.tensor().numpy()

    input_dict = {
        "input": input,
        "tensor1": tensor1,
        "tensor2": tensor2,
        "value": value
    }
    
    list_of_inputs.append(copy.deepcopy(input_dict))

    # 3 or more inputs like this
    
    return list_of_inputs

generated_inputs["torch.addcmul"] = addcmul_inputs()
```
""",
        "tf": """
```python
import tensorflow as tf
import copy

def tf_sets_difference_inputs():
    list_of_inputs = []
    # Input 1, valid
    a = tf.constant([[1, 2, 3], [4, 5, 6]])
    b = tf.constant([[2, 4, -6], [5, 7, 9]])
    aminusb = True
    validate_indices = True

    input_dict = {
        "a": a,
        "b": b,
        "aminusb": aminusb,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    # Input 2, valid
    a = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    b = tf.constant([[[2, 3], [4, 5]], [[6, 7], [8, 9]]])
    aminusb = False
    validate_indices = False

    input_dict = {
        "a": a,
        "b": b,
        "aminusb": aminusb,
        "validate_indices": validate_indices
    }
    list_of_inputs.append(copy.deepcopy(input_dict))

    return list_of_inputs

generated_inputs["tf.sets.difference"] = tf_sets_difference_inputs()
"""
    }
    doc = extract_function_info(fetch_documentation(api), api) if lib == "torch" else get_doc_tf(api)
    signature = get_signature(api, lib=lib, suffix=suffix)
    prefix = f'This is the documentation for the function {api}:\n\n"{doc.encode('ascii', errors='ignore').decode()}"\n\n' if doc else ""
    key = api if suffix == 0 else f"{api}_{suffix}"
    with open(f"{CUR_DIR}/prompt_input_gen.md", "r", encoding="utf-8") as file:
        prompt = file.read()
        prompt = prompt.replace("{api}", api)
        prompt = prompt.replace("{key}", key)
        prompt = prompt.replace("{signature}", str(signature))
        prompt = prompt.replace("{examples}", examples[lib])
    return prefix + prompt

def save_and_run_code(api, code, suffix=0, lib="torch"):
    key = api if suffix == 0 else f"{api}_{suffix}"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    validity_checker_code = f"""
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

{code}

def check_valid(api, list_of_inputs, lib="{lib}", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if '{key}' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key '{key}'.")

check_valid('{api}', generated_inputs['{key}'], lib="{lib}", suffix={suffix})
"""
    module_name = f'{api.replace(".", "_")}_{suffix}'
    filepath = f"{CUR_DIR}/inputs/{module_name}.py"
    with open(filepath, 'w') as f:
        f.write(validity_checker_code)
    
    try:
        # Run the generated file with a timeout of 30 sec just in case
        result = subprocess.run(['python', '-m', f'llm.inputs.{module_name}'], capture_output=True, text=True, timeout=30)
        
        return result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return "", "Timeout: Execution could not be completed in 30 seconds."

def retry_prompt(error):
    prompt = f"""Error faced during execution: {error}.
Please fix the error and retry the input generation. Only provide the code, skip any other text. Do not include verbose comments inside code. If you feel like you have added too many inputs and some of them are causing validity errors, remove them. If you do not feel confident about the error, try to generate a new input and delete the old one.
    """
    return prompt

def wrapper_parallel(variation):
    api, suffix = get_api_suffix(variation)
    if not "." in api:
        return "" 
    lib = api.split(".")[0]
    code = generate_inputs(api, suffix=suffix, lib=lib)
    return code

def generate_inputs(api, suffix=0, max_attempts=5, lib="torch"):
    # Set up logging
    logfile = os.path.join(CUR_DIR, "logs", f"{api}_{suffix}_input_gen.log")
    file_handler = logging.FileHandler(logfile, mode="w")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(asctime)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    logging.basicConfig(handlers=[file_handler])
    logger = logging.getLogger("generate_valid_inputs")

    model = "gemini-2.5-pro"
    gemini_key = os.getenv("gemini_key")

    logger.info(f"[{api}] [Suffix: {suffix}].\n\n")
    # time.sleep(6)
    client = genai.Client(api_key=gemini_key)
    chat = client.chats.create(model=model)
    try:
        prompt = get_prompt(api, lib=lib, suffix=suffix)
        logger.info(f"[Prompt]\n\n{prompt}\n\n")
        response = chat.send_message(prompt)
        logger.info(f"[Response]\n\n{response.text}\n\n")
    except genai.errors.ServerError as ge:
        logger.warning(f"Server overloaded. Error: {str(ge)}")
        logger.warning(f"Waiting 10 seconds before retrying...")
        time.sleep(10)
        return generate_inputs(api, suffix=suffix, max_attempts=max_attempts, lib=lib)
    except Exception as e:
        logger.warning(f"Error while sending message to Gemini API: {e}")
        logger.warning(f"Waiting 10 seconds before retrying...")
        time.sleep(10)
        return generate_inputs(api, suffix=suffix, max_attempts=max_attempts, lib=lib)
    logger.info("Got response from Gemini API.")
    code = extract_code_from_response(response.text)    
    output, error = save_and_run_code(api, code, suffix=suffix, lib=lib)
    logger.info(f"[Output]\n\n{output}\n\n")
    logger.info(f"[Error]\n\n{error}\n\n") if error else logger.info("No error\n\n")
    attempt = 0
    to_return = [0] * max_attempts
    
    while not output.endswith("Valid"):
        logger.info(f"Attempt {attempt + 1}:\n{error}")
        to_return[attempt] = 1
        logger.info("Retrying code generation after 6 seconds...")
        # time.sleep(6)
        prompt = retry_prompt(error)
        logger.info(f"[Retry Prompt]\n\n{prompt}\n\n")
        try:
            response = chat.send_message(prompt)
        except genai.errors.ServerError as ge:
            logger.warning(f"Server overloaded. Error: {str(ge)}")
            logger.warning(f"Waiting 10 seconds before retrying...")
            time.sleep(10)
            continue
        except Exception as e:
            logger.warning(f"Error while sending message to Gemini API: {e}")
            logger.warning(f"Waiting 10 seconds before retrying...")
            time.sleep(10)
            continue
        logger.info("Got response from Gemini API.")
        logger.info(f"[Response]\n\n{response.text}\n\n")
        code = extract_code_from_response(response.text)
        output, error = save_and_run_code(api, code, suffix=suffix, lib=lib)
        logger.info(f"[Output]\n\n{output}\n\n")
        logger.info(f"[Error]\n\n{error}\n\n") if error else logger.info("No error\n\n")
        attempt += 1
        
        if attempt >= max_attempts:
            logger.error(f"Max attempts reached. Exiting.\n\n")
            break
        
    if output.endswith("Valid"):
        logger.info(f"API: {api} Suffix: {suffix} | Input generated successfully.\n\n")
        if code is not None:
            code = code.replace("generated_inputs = {}", "")
            return code
        else:
            logger.info("No code to write to valid_inputs.py.")
            return ""
    else:
        logger.info(f"API: {api} Suffix: {suffix} | Input generation failed after {max_attempts} attempts.\n\n")
    return ""

def main():
    lib = sys.argv[1] if len(sys.argv) > 1 else "torch"
    n_procs = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    variations = read_file_in_root(f"{lib}_variations.txt")
    os.makedirs(f"{CUR_DIR}/logs", exist_ok=True)
    
    if lib == "torch":
        valid_inputs = valid_inputs_torch
    elif lib == "tf":
        valid_inputs = valid_inputs_tf
    else:
        raise ValueError(f"Invalid library: {lib}")
    
    generated_inputs = valid_inputs.generated_inputs
    existing_inputs = set(generated_inputs.keys())
    variations = [v for v in variations if v not in existing_inputs]
    durations = []
    
    filtered_variations = []
    for variation in variations:
        if variation in generated_inputs:
            print(f"{bcolors.OKGREEN}Skipping {variation} as it is already generated.{bcolors.ENDC}")
            continue
        filtered_variations.append(variation)
    
    total = len(filtered_variations)
    print(f"Existing inputs: {len(existing_inputs)} | Variations to generate: {total}\n")
    idx = 0
    start = time.time()
    with Pool(processes=n_procs) as pool:
        for code in pool.imap_unordered(wrapper_parallel, filtered_variations):
            if code:
                with open(f"{CUR_DIR}/valid_inputs_{lib}.py", "a") as fv:
                    fv.write(code + "\n\n")
            else:
                print(f"{bcolors.WARNING}Code generation failed.{bcolors.ENDC}")
            
            durations.append(time.time()-start)
            start = time.time()
            idx += 1
            print(f"{bcolors.OKGREEN}Done with {idx}/{total} | ETR: {(total-idx)*np.mean(durations):.2f}s{bcolors.ENDC}")
if __name__ == "__main__":
    main()