from google import genai
import os, time
from llm.llm_utils import (
    OAChatWrapper,
    fetch_documentation,
    extract_code_from_response,
    extract_function_info,
    Response,
    collect_token_usage,
)
from utils.misc import read_file_in_root
from utils.new_api_utils import get_doc_tf, get_api_suffix
import sys
import logging

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)

def format_usage(usage):
    if not usage:
        return "Token usage: input=n/a, output=n/a, total=n/a"
    input_tokens = usage.get("input_tokens", "n/a")
    output_tokens = usage.get("output_tokens", "n/a")
    total_tokens = usage.get("total_tokens", "n/a")
    return f"Token usage: input={input_tokens}, output={output_tokens}, total={total_tokens}"

def get_prompt(api, lib="torch"):
    examples = {
        "torch": """
```python
signatures["torch.scatter_add"] = {
    "args": {
        "input": "tensor",
        "dim": "integer",
        "index": "tensor",
        "src": "tensor"
    },
    "kwargs": {},
    "inner": {},
}
signatures["torch.add"] = {
    "args": {
        "input": "tensor",
        "other": "tensor"
    },
    "kwargs": {
        "alpha": "float",
        "out": "tensor"
    },
    "inner": {},
}
signatures["torch.nn.ReflectionPad1d_1"] = {
    "args": {
        "padding": "integer"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor",
            "other": "tensor"
        },
        "kwargs": {}
    }
}
signatures["torch.nn.ReflectionPad1d_2"] = {
    "args": {
        "padding": "tuple"
    },
    "kwargs": {},
    "inner": {
        "args": {
            "input": "tensor",
            "other": "tensor"
        },
        "kwargs": {}
    }
}
```
""",
        "tf": """
```python
signatures["tf.abs"] = {
    "args": {
        "x": "tensor"
    },
    "kwargs": {
        "name": "string"
    },
    "inner": {}
}
signatures["tf.signal.rfft"] = {
    "args": {
        "input_tensor": "tensor"
    },
    "kwargs": {
        "fft_length": "integer", # Tensor of type int32 and shape [1] is an integer
        "name": "string"
    },
    "inner": {}
}
```
"""
    }

    if lib == "torch":
        doc = extract_function_info(fetch_documentation(api), api)
    else:
        doc = get_doc_tf(api)
        
    prefix = f'This is the documentation for the function {api}:\n\n"{doc.encode('ascii', errors='ignore').decode()}"\n\n' if doc else ""
    if lib == "torch":
        callables = f'This api likely returns a function, look for the parameters that can be passed to the function returned by this api. Hint: very often this information can be found under the "Shape:" section of the documentation.' if api.split('.')[-1][0].isupper() else 'This api likely does not return a function, check if that is true. If so, `inner` should be empty. Otherwise add the signature for the inner call.'
    elif lib == "tf":
        callables = ""
    with open(f"{CUR_DIR}/prompt_signature_gen.md", "r", encoding="utf-8") as file:
        prompt = file.read()
        prompt = prompt.replace("{api}", api)
        prompt = prompt.replace("{callables}", callables)
        prompt = prompt.replace("{examples}", examples[lib])
    return prefix + prompt

def save_sig(sig, lib, llm="gemini"):
    filepath = f"{CUR_DIR}/{llm}/{lib}_signatures.py"
    with open(filepath, 'a') as f:
        f.write(sig + '\n')

def generate_signatures(api, lib="torch", llm="gemini"):
    print("Running signature generation after 6 seconds...")
    time.sleep(6)
    
    if llm == "gemini":
        model = "gemini-2.0-flash"
        gemini_key = os.getenv("gemini_key")
        client = genai.Client(api_key=gemini_key)
        chat = client.chats.create(model=model)
    elif llm == "openai":
        chat = OAChatWrapper()
    else:
        raise ValueError("llm must be either 'gemini' or 'openai'")

    try:
        prompt = get_prompt(api, lib=lib)
    except Exception as e:
        print(f"\nGenerating prompt for {api} faced exception.\n{e.__class__.__name__}: {str(e)}\n")
        with open(f"{CUR_DIR}/failed_sig_{lib}.txt", "a") as f:
            f.write(f"{api}\n")
        return
    logger.info(f"[Prompt]\n\n{prompt}\n\n")
    
    raw_response = chat.send_message(prompt)
    if isinstance(raw_response, Response):
        response_obj = raw_response
    else:
        response_text = getattr(raw_response, "text", None) or getattr(raw_response, "output_text", "")
        usage_metadata = collect_token_usage(getattr(raw_response, "usage_metadata", None))
        response_obj = Response(text=response_text, usage=usage_metadata)

    logger.info(f"[Response]\n\n{response_obj.text}\n\n")
    logger.info(f"[Usage]\n\n{format_usage(response_obj.usage)}\n\n")
    sig = extract_code_from_response(response_obj.text, llm=llm)
    print(f"Got response from {llm} API:\n{sig}")
    if sig is not None:
        save_sig(sig, lib=lib, llm=llm)
    else:
        with open(f"{CUR_DIR}/{llm}/needs_sig_{lib}.txt", "a") as f:
            f.write(f"{api}\n")

def main():
    lib = sys.argv[1] if len(sys.argv) > 1 else "torch"
    llm = sys.argv[2] if len(sys.argv) > 2 else "gemini"

    logfile = f"{CUR_DIR}/{llm}/signature_creation_{lib}.log"
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,                                     # Minimum log level
        format='%(message)s',                                   # Log format
        filename=logfile,                                       # Log file path
        filemode="a"                                            # Append/Write mode
    )

    apis = read_file_in_root(f"{lib}_apis.txt")

    if llm == "gemini":
        from llm.gemini.tf_signatures import signatures as tf_signatures
        from llm.gemini.torch_signatures import signatures as torch_signatures
    elif llm == "openai":
        from llm.openai.tf_signatures import signatures as tf_signatures
        from llm.openai.torch_signatures import signatures as torch_signatures
    else:
        raise ValueError("llm must be either 'gemini' or 'openai'")

    signatures = torch_signatures if lib == "torch" else tf_signatures
    completed = set()
    for variation in signatures.keys():
        api, suffix = get_api_suffix(variation)
        completed.add(api)

    for api in apis:
        if api in completed:
            print(f"Signature exists for {api}. Skipping...")
            continue
        print(f"\nGenerating valid signatures for {api}...\n")
        generate_signatures(api, lib=lib, llm=llm)
        
if __name__ == "__main__":
    main()
