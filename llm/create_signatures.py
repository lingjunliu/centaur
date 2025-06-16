from google import genai
import os, time
from llm.create_driver import fetch_documentation, extract_code_from_response, extract_function_info
from utils.misc import read_file_in_root

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def get_api_basename(torch_api):
    api = None
    with open(f"{CUR_DIR}/supported.csv", "r") as f:
        for line in f.readlines():
            tokens = line.strip().split(",")
            if tokens[1] == torch_api:
                api = tokens[0]
                break
    return api

def get_prompt(api):
    doc = extract_function_info(fetch_documentation(api), api)
    prefix = f'This is the documentation for the function {api}:\n\n"{doc.encode('ascii', errors='ignore').decode()}"\n\n' if doc else ""
    callables = f'This api likely returns a function, look for the parameters that can be passed to the function returned by this api. Hint: very often this information can be found under the "Shape:" section of the documentation.' if api.split('.')[-1][0].isupper() else 'This api likely does not return a function, check if that is true. If so, `inner` should be empty. Otherwise add the signature for the inner call.'
    with open(f"{CUR_DIR}/prompt_signature_gen.md", "r", encoding="utf-8") as file:
        prompt = file.read()
        prompt = prompt.replace("{api}", api)
        prompt = prompt.replace("{callables}", callables)
    return prefix + prompt

def save_sig(sig):
    filepath = f"{CUR_DIR}/signatures.py"
    with open(filepath, 'a') as f:
        f.write(sig + '\n')

def generate_signatures(api):
    model = "gemini-2.0-flash"
    gemini_key = os.getenv("gemini_key")

    print("Running signature generation after 6 seconds...")
    time.sleep(6)
    client = genai.Client(api_key=gemini_key)
    chat = client.chats.create(model=model)
    response = chat.send_message(get_prompt(api))
    sig = extract_code_from_response(response.text)
    print(f"Got response from Gemini API:\n{sig}")
    if sig is not None:
        save_sig(sig)
    else:
        with open(f"{CUR_DIR}/needs_sig.txt", "a") as f:
            f.write(f"{api}\n")

def main():
    torch_apis = read_file_in_root("torch_apis.txt")
    
    for torch_api in torch_apis:
        print(f"\nGenerating valid signatures for {torch_api}...\n")
        generate_signatures(torch_api)
        
if __name__ == "__main__":
    main()