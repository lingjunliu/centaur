from google import genai
import os, re, subprocess, requests, time
from bs4 import BeautifulSoup
from llm.get_api_list import update_apis
from llm.signatures_old import signatures

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

def extract_code_from_response(response):
    # Use regular expression to find the code block within the markdown
    try:
        code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            return code_match.group(1)
    except Exception as e:
        print(f"Error extracting code: {e}")
    return ""

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

def fetch_documentation(function_name):
    base_url = "https://pytorch.org/docs/stable/generated/"
    function_url = base_url + function_name + ".html"
    response = requests.get(function_url)
    if response.status_code == 200:
        return response.text
    else:
        return None

def clean_text(text):
    #remove excessive newlines
    return ' '.join(text.split())

def extract_function_info(html_content, function_name):
    if not html_content:
        return None
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Try finding the section using different possible patterns
    section_id = function_name.replace('.', '-').lower()
    doc_content = soup.find('div', {'class': 'section', 'id': section_id})
    
    if not doc_content:
        section_id = function_name.replace('.', '-')
        doc_content = soup.find('div', {'class': 'section', 'id': section_id})
    
    if not doc_content:
        # If specific section ID is not found, use a more general approach
        doc_content = soup.find('dl', {'class': 'py class'})

    if doc_content:
        # Extract text while avoiding duplicates
        lines = []
        seen_lines = set()
        for element in doc_content.find_all(['p', 'pre', 'code', 'dd', 'dt', 'ul', 'li', 'h1', 'h2', 'h3']):
            cleaned_line = clean_text(element.get_text())
            if cleaned_line not in seen_lines:
                lines.append(cleaned_line)
                seen_lines.add(cleaned_line)
        return '\n'.join(lines)
    else:
        return None

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