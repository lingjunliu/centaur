import os, re

base = "rulegen/claude-tf/rules-tf"
pattern = re.compile(r"Token usage:\s*input=(\d+), output=(\d+), total=(\d+)")

input_total = output_total = total_total = 0
entries = logs = 0

for entry in os.scandir(base):
    if not entry.is_dir():
        continue
    log_path = os.path.join(entry.path, "log-rulegen")
    if not os.path.isfile(log_path):
        continue
    logs += 1
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                entries += 1
                input_total += int(m.group(1))
                output_total += int(m.group(2))
                total_total += int(m.group(3))

print("logs:", logs)
print("entries:", entries)
print("input_total:", input_total)
print("output_total:", output_total)
print("total_total:", total_total)
