from google import genai
import os, sys

def get_token_count(client, model: str, prompt: str) -> int:
    total_tokens = client.models.count_tokens(model=model, contents=prompt)
    return total_tokens.total_tokens

def parse_log(log_path: str) -> int:
    prompts={}
    responses={}
    cur_prompt = None
    cur_response = None
    cur_api = ""
    prompt = True
    with open(log_path, 'r') as log_file:
        for line in log_file:
            if line.strip() == "[Prompt]":
                prompt = True
                cur_prompt = ""
                if cur_response is not None:
                    responses[cur_api] = cur_response
                    cur_response = None
            elif line.strip() == "[Response]":
                prompt = False
                cur_response = ""
                if cur_prompt is not None:
                    prompts[cur_api] = cur_prompt
                    cur_prompt = None
            elif line.strip() in ["[Output]", "[Error]"]:
                cur_response = None
                cur_prompt = None
                continue
            elif prompt and cur_prompt is not None:
                cur_prompt += line
                if line.startswith("This is the documentation for the function"):
                    # e.g. This is the documentation for the function torch.nn.AdaptiveLogSoftmaxWithLoss:
                    cur_api = line.split()[-1][:-1]
            elif cur_response is not None:
                cur_response += line
            else:
                continue

    return prompts, responses


def main():
    log_path = sys.argv[1]
    prompts, responses = parse_log(log_path)
    gemini_key = os.getenv("gemini_key")
    client = genai.Client(api_key=gemini_key)

    tokens_file = os.path.join(os.path.dirname(log_path), f"{os.path.basename(log_path)}_tokens.csv")

    with open(tokens_file, 'w') as f:
        # api	input_tokens	output_tokens	total_tokens
        f.write("api,input_tokens,output_tokens,total_tokens\n")
        for api in prompts:
            if api not in responses:
                continue
            input_tokens = get_token_count(client, model="gemini-2.0-flash", prompt=prompts[api])
            output_tokens = get_token_count(client, model="gemini-2.0-flash", prompt=responses[api])
            total_tokens = input_tokens + output_tokens
            f.write(f"{api},{input_tokens},{output_tokens},{total_tokens}\n")    

if __name__ == "__main__":
    main()
