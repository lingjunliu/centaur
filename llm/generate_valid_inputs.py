from google import genai
import os, subprocess, time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
from utils.new_api_utils import get_n_variations, get_signature, get_doc_tf, get_api_suffix
from utils.misc import read_file_in_root, bcolors
from llm.llm_utils import (
    OAChatWrapper,
    fetch_documentation,
    extract_code_from_response,
    extract_function_info,
    Response,
    collect_token_usage,
)
import logging
import sys

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)

def format_usage(usage):
    if not usage:
        return "Token usage: input=n/a, output=n/a, total=n/a"
    input_tokens = usage.get("input_tokens", "n/a")
    output_tokens = usage.get("output_tokens", "n/a")
    total_tokens = usage.get("total_tokens", "n/a")
    return f"Token usage: input={input_tokens}, output={output_tokens}, total={total_tokens}"

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

tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)

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

def save_and_run_code(api, code, suffix=0, lib="torch", llm="gemini"):
    key = api if suffix == 0 else f"{api}_{suffix}"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf_snippet = """
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(42)
""" if lib == "tf" else ""
    
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

{tf_snippet}
check_valid('{api}', generated_inputs['{key}'], lib="{lib}", suffix={suffix})
"""
    module_name = f'{api.replace(".", "_")}_{suffix}'
    filepath = f"{CUR_DIR}/{llm}/inputs/{module_name}.py"
    with open(filepath, 'w') as f:
        f.write(validity_checker_code)
    
    try:
        # Run the generated file with a timeout of 30 sec just in case
        result = subprocess.run(['python', '-m', f'llm.{llm}.inputs.{module_name}'], capture_output=True, text=True, timeout=30)
        
        return result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        print(f"Execution of {module_name} timed out.")
        return "", "Timeout: Execution could not be completed in 30 seconds."

def retry_prompt(error):
    prompt = f"""Error faced during execution: {error}.
Please fix the error and retry the input generation. Only provide the code, skip any other text. Do not include verbose comments inside code. If you feel like you have added too many inputs and some of them are causing validity errors, remove them. If you do not feel confident about the error, try to generate a new input and delete the old one.
    """
    return prompt

def generate_inputs(api, suffix=0, max_attempts=5, lib="torch", llm="gemini"):
    print(f"{bcolors.OKBLUE}Running code generation for {api} with suffix {suffix} after 6 seconds...{bcolors.ENDC}")
    logger.info(f"[{api}] [Suffix: {suffix}].\n\n")
    time.sleep(6)
    
    if llm == "gemini":
        model = "gemini-3.5-flash"
        gemini_key = os.getenv("gemini_key")
        client = genai.Client(api_key=gemini_key)
        chat = client.chats.create(model=model)
    elif llm == "openai":
        chat = OAChatWrapper()
    else:
        raise ValueError("llm must be either 'gemini' or 'openai'")
    
    try:
        prompt = get_prompt(api, lib=lib, suffix=suffix)
        logger.info(f"[Prompt]\n\n{prompt}\n\n")
    except Exception as e:
        print(f"\nGenerating prompt for {api} faced exception.\n{e.__class__.__name__}: {str(e)}\n")
        with open(f"{CUR_DIR}/{llm}/failed_input_gen_{lib}.txt", "a") as f:
            f.write(f"{api},{suffix}\n")
        return [api, api] + [1]*max_attempts

    try:
        raw_response = chat.send_message(prompt)
        if isinstance(raw_response, Response):
            response_obj = raw_response
        else:
            response_text = getattr(raw_response, "text", None) or getattr(raw_response, "output_text", "")
            usage_metadata = collect_token_usage(getattr(raw_response, "usage_metadata", None))
            response_obj = Response(text=response_text, usage=usage_metadata)

        logger.info(f"[Response]\n\n{response_obj.text}\n\n")
        logger.info(f"[Usage]\n\n{format_usage(response_obj.usage)}\n\n")
    except genai.errors.ServerError as ge:
        print(f"{bcolors.WARNING}Server overloaded. Error: {str(ge)}{bcolors.ENDC}")
        print(f"{bcolors.WARNING}Waiting 10 seconds before retrying...{bcolors.ENDC}")
        time.sleep(10)
        return generate_inputs(api, suffix=suffix, max_attempts=max_attempts, lib=lib, llm=llm)
    except Exception as e:
        print(f"{bcolors.FAIL}Error while sending message to {llm} API: {e}{bcolors.ENDC}")
        print(f"{bcolors.WARNING}Waiting 10 seconds before retrying...{bcolors.ENDC}")
        time.sleep(10)
        return generate_inputs(api, suffix=suffix, max_attempts=max_attempts, lib=lib, llm=llm)
    print(f"Got response from {llm} API.")
    code = extract_code_from_response(response_obj.text, llm=llm)
    output, error = save_and_run_code(api, code, suffix=suffix, lib=lib, llm=llm)
    logger.info(f"[Output]\n\n{output}\n\n")
    logger.info(f"[Error]\n\n{error}\n\n") if error else logger.info("No error\n\n")
    attempt = 0
    to_return = [0] * max_attempts
    
    while not output.endswith("Valid"):
        print(f"{bcolors.OKBLUE}Attempt {attempt + 1}:{bcolors.ENDC}\n{error}")
        to_return[attempt] = 1
        print("Retrying code generation after 6 seconds...")
        # time.sleep(6)
        prompt = retry_prompt(error)
        logger.info(f"[Retry Prompt]\n\n{prompt}\n\n")
        try:
            raw_response = chat.send_message(prompt)
        except genai.errors.ServerError as ge:
            print(f"{bcolors.WARNING}Server overloaded. Error: {str(ge)}{bcolors.ENDC}")
            print(f"{bcolors.WARNING}Waiting 10 seconds before retrying...{bcolors.ENDC}")
            time.sleep(10)
            continue
        except Exception as e:
            print(f"{bcolors.FAIL}Error while sending message to Gemini API: {e}{bcolors.ENDC}")
            print(f"{bcolors.WARNING}Waiting 10 seconds before retrying...{bcolors.ENDC}")
            time.sleep(10)
            continue
        if isinstance(raw_response, Response):
            response_obj = raw_response
        else:
            response_text = getattr(raw_response, "text", None) or getattr(raw_response, "output_text", "")
            usage_metadata = collect_token_usage(getattr(raw_response, "usage_metadata", None))
            response_obj = Response(text=response_text, usage=usage_metadata)

        print(f"Got response from {llm} API.")
        logger.info(f"[Response]\n\n{response_obj.text}\n\n")
        logger.info(f"[Usage]\n\n{format_usage(response_obj.usage)}\n\n")
        code = extract_code_from_response(response_obj.text, llm=llm)
        output, error = save_and_run_code(api, code, suffix=suffix, lib=lib, llm=llm)
        logger.info(f"[Output]\n\n{output}\n\n")
        logger.info(f"[Error]\n\n{error}\n\n") if error else logger.info("No error\n\n")
        attempt += 1
        
        if attempt >= max_attempts:
            print(f"{bcolors.FAIL}Max attempts reached. Exiting.{bcolors.ENDC}")
            logger.info("Max attempts reached. Exiting.\n\n")
            break
        
    if output.endswith("Valid"):
        print("\nInput generated successfully.")
        logger.info(f"API: {api} Suffix: {suffix} | Input generated successfully.\n\n")
        if code is not None:
            code = code.replace("generated_inputs = {}", "")
            with open(f"{CUR_DIR}/{llm}/valid_inputs_{lib}.py", "a") as fv:
                fv.write(code + "\n\n")
        else:
            print("No code to write to valid_inputs.py.")
    else:
        print(f"\nInput generation failed after {max_attempts} attempts.")
        logger.info(f"API: {api} Suffix: {suffix} | Input generation failed after {max_attempts} attempts.\n\n")
    return [api, api] + to_return

def main():
    lib = sys.argv[1] if len(sys.argv) > 1 else "torch"
    llm = sys.argv[2] if len(sys.argv) > 2 else "gemini"

    variations_file = f"{CUR_DIR}/{llm}/{lib}_variations.txt"
    variations = []
    if os.path.exists(variations_file):
        with open(variations_file, "r") as f:
            variations = [line.strip() for line in f.readlines()]

    logfile = f"{CUR_DIR}/{llm}/input_generation_{lib}.log"
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,                                     # Minimum log level
        format='%(message)s',                                   # Log format
        filename=logfile,                                       # Log file path
        filemode="a"                                            # Append/Write mode
    )
    
    if llm == "gemini":
        if lib == "torch":
            import llm.gemini.valid_inputs_torch as valid_inputs
        elif lib == "tf":
            import llm.gemini.valid_inputs_tf as valid_inputs
        else:
            raise ValueError("lib must be either 'torch' or 'tf'")
    elif llm == "openai":
        if lib == "torch":
            import llm.openai.valid_inputs_torch as valid_inputs
        elif lib == "tf":
            import llm.openai.valid_inputs_tf as valid_inputs
        else:
            raise ValueError("lib must be either 'torch' or 'tf'")
    else:
        raise ValueError("llm must be either 'gemini' or 'openai'")
    
    generated_inputs = valid_inputs.generated_inputs
    existing_inputs = set(generated_inputs.keys())
    variations = [v for v in variations if v not in existing_inputs]
    total = len(variations)
    durations = []
    print(f"Existing inputs: {len(existing_inputs)} | Variations to generate: {total}\n")
    
    for idx, variation in enumerate(variations):
        if variation in generated_inputs:
            print(f"{bcolors.OKGREEN}Skipping {variation} as it is already generated.{bcolors.ENDC}")
            continue
        api, suffix = get_api_suffix(variation)
        start = time.time()
        print(f"\nGenerating valid inputs for {api} with suffix {suffix}...\n")
        result = generate_inputs(api, suffix=suffix, lib=lib, llm=llm)
        with open(f"{CUR_DIR}/{llm}/inputs.csv", "a") as f:
            f.write(",".join(map(str, result)) + "\n")
        
        durations.append(time.time()-start)
        print(f"{bcolors.OKGREEN}Done with {idx+1}/{total} | ETR: {(total-idx-1)*np.mean(durations):.2f}s{bcolors.ENDC}")
        
if __name__ == "__main__":
    main()
