from openai import OpenAI
import os, re, requests
from bs4 import BeautifulSoup

class Response:
    def __init__(self, text, usage=None):
        self.text = text
        self.usage = usage or {}

def collect_token_usage(usage_obj):
    """
    Normalize usage metadata into a dict with input/output/total tokens.
    """
    if usage_obj is None:
        return {}

    usage_data = None

    if isinstance(usage_obj, dict):
        usage_data = usage_obj
    else:
        for attr_name in ("model_dump", "dict"):
            attr = getattr(usage_obj, attr_name, None)
            if callable(attr):
                try:
                    usage_data = attr()
                    break
                except TypeError:
                    continue
        if usage_data is None:
            usage_data = {}
            for name in dir(usage_obj):
                if name.startswith("_"):
                    continue
                try:
                    value = getattr(usage_obj, name)
                except AttributeError:
                    continue
                if isinstance(value, (int, float)):
                    usage_data[name] = value

    def _coerce(names):
        for name in names:
            if name in usage_data and usage_data[name] is not None:
                value = usage_data[name]
                if isinstance(value, (int, float)):
                    return int(value)
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return value
        return None

    input_tokens = _coerce(("input_tokens", "prompt_tokens", "prompt_token_count"))
    output_tokens = _coerce(("output_tokens", "completion_tokens", "candidates_token_count"))
    total_tokens = _coerce(("total_tokens", "total_token_count"))

    usage = {}
    if input_tokens is not None:
        usage["input_tokens"] = input_tokens
    if output_tokens is not None:
        usage["output_tokens"] = output_tokens
    if total_tokens is not None:
        usage["total_tokens"] = total_tokens
    elif input_tokens is not None and output_tokens is not None:
        usage["total_tokens"] = input_tokens + output_tokens

    return usage

class OAChatWrapper:
    DEFAULT_MODEL = "gpt-5"
    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(self, model=None, base_url=None, api_key=None):
        env_model = os.getenv("MODEL_NAME")
        env_url = os.getenv("MODEL_URL")

        # Prioritize explicit arguments, then env vars, then defaults
        self.model = model or env_model or self.DEFAULT_MODEL
        configured_url = base_url or env_url or self.DEFAULT_BASE_URL
        normalized_url = configured_url.rstrip("/")

        # Determine which OpenAI API surface to use based on the configured URL
        if normalized_url.endswith("/chat/completions"):
            self._api_surface = "chat_completions"
            client_base_url = normalized_url.rsplit("/chat/completions", 1)[0]
        elif normalized_url.endswith("/completions"):
            self._api_surface = "completions"
            client_base_url = normalized_url.rsplit("/completions", 1)[0]
        else:
            self._api_surface = "responses"
            client_base_url = normalized_url

        if not client_base_url:
            client_base_url = self.DEFAULT_BASE_URL

        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=client_base_url,
        )
        self.previous_response_id = None
        self._messages = []  # Used when interacting via chat/completions

    def send_message(self, prompt):
        usage = {}
        if self._api_surface == "responses":
            if self.previous_response_id is None:
                openai_response = self.client.responses.create(
                    model=self.model,
                    input=[{"role": "user", "content": prompt}]
                )
            else:
                openai_response = self.client.responses.create(
                    model=self.model,
                    previous_response_id=self.previous_response_id,
                    input=[{"role": "user", "content": prompt}]
                )
            self.previous_response_id = openai_response.id
            response_text = openai_response.output_text or ""
            usage = collect_token_usage(getattr(openai_response, "usage", None))
        elif self._api_surface == "chat_completions":
            self._messages.append({"role": "user", "content": prompt})
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=self._messages,
            )
            message_content = completion.choices[0].message.content
            if isinstance(message_content, list):
                response_text = "".join(part if isinstance(part, str) else part.get("text", "") for part in message_content)
            else:
                response_text = message_content or ""
            self._messages.append({"role": "assistant", "content": response_text})
            usage = collect_token_usage(getattr(completion, "usage", None))
        elif self._api_surface == "completions":
            # Fallback for legacy completion endpoints – maintain a running prompt.
            self._messages.append(prompt)
            completion = self.client.completions.create(
                model=self.model,
                prompt="\n\n".join(self._messages),
            )
            response_text = completion.choices[0].text or ""
            self._messages.append(response_text)
            usage = collect_token_usage(getattr(completion, "usage", None))
        else:
            raise ValueError(f"Unsupported OpenAI API surface: {self._api_surface}")

        return Response(text=response_text, usage=usage)
    
def extract_code_from_response(response, llm="gemini"):
    # Use regular expression to find the code block within the markdown
    try:
        code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            return code_match.group(1)
    except Exception as e:
        print(f"Error extracting code: {e}")

    if llm == "gemini":
        return ""
    elif llm == "openai":
        return response

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
