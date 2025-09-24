from google import genai
import os, subprocess, time
from llm.get_api_list import update_apis
from llm.signatures_old import signatures
from llm.llm_utils import fetch_documentation, extract_code_from_response, extract_function_info

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def get_prompt(api):
    doc = extract_function_info(fetch_documentation(api), api)
    prefix = f'This is the documentation for the function {api}:\n\n"{doc.encode('ascii', errors='ignore').decode()}"\n\n' if doc else ""
    if api in signatures:
        prefix += f"""This is the signature for this file:
        {signatures[api]}
        """
    with open(f"{CUR_DIR}/prompt.md", "r", encoding="utf-8") as file:
        prompt = file.read()
        prompt = prompt.replace("{api}", api)
    return prefix + prompt

def save_and_run_code(filename, code):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    filepath = f"{CUR_DIR}/drivers/{filename}.py"
    with open(filepath, 'w') as f:
        f.write(code)
    
    try:
        # Run the generated file with a timeout of 30 sec just in case
        result = subprocess.run(['python', '-m', f'llm.drivers.{filename}'], capture_output=True, text=True, timeout=30)
        # print(result.stdout)
        # print(result.stderr)
        error = result.stderr.strip()
        if "UNKNOWN ERROR (303)" in result.stderr:
            error = error.split("UNKNOWN ERROR (303)")[1]
        
        return result.stdout.strip(), error.strip()
    except subprocess.TimeoutExpired:
        print(f"Execution of {filename} timed out.")
        return "", "Timeout: Execution could not be completed in 30 seconds."

def retry_prompt(error):
    prompt = f"""Error Faced during execution: {error}.
Please fix the error and retry the code generation. Only provide the code, skip any other text. Do not include verbose comments inside code.
    """
    return prompt

def driver_to_api(driver):
    """
        Takes a driver name and returns the corresponding torch API name.
    """
    with open(f"{CUR_DIR}/drivers_to_api.csv", "r") as f:
        for line in f.readlines():
            tokens = line.strip().split(",")
            if tokens[0] == driver:
                return tokens[1]
    return None

def generate_driver(api, max_attempts=5):
    api_basename = api.split(".")[-1]
    existing_api = driver_to_api(api_basename)
    counter = 1
    while existing_api is not None and api != existing_api:
        # There is already a drriver with the same name for a different API
        tokens = api.split(".")
        if len(tokens) > 2:
            api_basename = tokens[-2] + "_" + api_basename
        else:
            api_basename = api_basename + f"_{counter}"
            counter += 1
        existing_api = driver_to_api(api_basename)

    to_return = [0] * max_attempts
    
    model = "gemini-2.0-flash"
    gemini_key = os.getenv("gemini_key")

    print("Running code generation after 6 seconds...")
    time.sleep(6)
    client = genai.Client(api_key=gemini_key)
    chat = client.chats.create(model=model)
    response = chat.send_message(get_prompt(api))
    print("Got response from Gemini API.")
    code = extract_code_from_response(response.text)
    output, error = save_and_run_code(api_basename, code)
    attempt = 0
    
    while error > "" and not output.endswith("Success"):
        print(f"Attempt {attempt + 1}: Error occurred.\n\n{error}")
        to_return[attempt] = 1
        print("Retrying code generation after 6 seconds...")
        time.sleep(6)
        response = chat.send_message(retry_prompt(error))
        print("Got response from Gemini API.")
        code = extract_code_from_response(response.text)
        output, error = save_and_run_code(api_basename, code)
        attempt += 1
        
        if attempt >= max_attempts:
            print("Max attempts reached. Exiting.")
            break
        
    if error == "" or output.endswith("Success"):
        print("\nCode executed successfully.")
    else:
        print("\nCode execution failed after multiple attempts.")
        
    return [api_basename, api] + to_return

def main():
    with open(f"{CUR_DIR}/needs_driver.txt", "r") as f:
        apis = [line.strip() for line in f.readlines()]
    
    for api in apis:
        print(f"\n\nGenerating driver for {api}...\n\n")
        result = generate_driver(api)
        with open(f"{CUR_DIR}/drivers.csv", "a") as f:
            f.write(",".join(map(str, result)) + "\n")
    
    update_apis()
        
if __name__ == "__main__":
    main()