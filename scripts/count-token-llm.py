import os, re

models = [
    "claude-tf",
    "claude-torch",
    "gemma3-tf",
    "gemma3-torch",
    "qwen-tf",
    "qwen-torch",
]
pattern = re.compile(r"Token usage:\s*input=(\d+), output=(\d+), total=(\d+)")

for model in models:
    base = os.path.join("llm", model)
    input_total = output_total = total_total = entries = 0
    log_files = []
    for root, _, files in os.walk(base):
        for name in files:
            if name.endswith(".log"):
                log_files.append(os.path.join(root, name))
    for log_path in log_files:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    entries += 1
                    input_total += int(m.group(1))
                    output_total += int(m.group(2))
                    total_total += int(m.group(3))
    print(
        f"{model}: logs={len(log_files)}, entries={entries}, "
        f"input={input_total}, output={output_total}, total={total_total}"
    )
