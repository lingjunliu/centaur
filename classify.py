import subprocess, re, sys

# Library to classify: tf or torch (default tf). Determines rules-<lib>/ subtree.
lib = sys.argv[1] if len(sys.argv) > 1 else "tf"
rules_dir = f"rules-{lib}/"

# Diff of all modified tracked files under rules-<lib>/, no context lines.
diff = subprocess.run(
    ["git", "diff", "--unified=0", "--", rules_dir],
    capture_output=True, text=True,
).stdout

def norm(line):
    """Normalize a changed line by sorting keys inside any {...} dict literal,
    so two lines that differ only in key order become identical."""
    line = line.rstrip("\n")
    m = re.search(r"\{(.*)\}", line)
    if not m:
        return line
    inner = m.group(1)
    parts = [p.strip() for p in inner.split(",") if p.strip()]
    parts_sorted = sorted(parts, key=lambda p: p.split(":", 1)[0].strip())
    return line[:m.start(1)] + ", ".join(parts_sorted) + line[m.end(1):]

files = {}
cur = None
for line in diff.splitlines():
    if line.startswith("+++ b/"):
        cur = line[6:]
        files[cur] = {"add": [], "del": []}
    elif line.startswith("+") and not line.startswith("+++"):
        files[cur]["add"].append(line[1:])
    elif line.startswith("-") and not line.startswith("---"):
        files[cur]["del"].append(line[1:])

reorder_only, substantive = [], []
for f, d in files.items():
    na = sorted(norm(x) for x in d["add"])
    nd = sorted(norm(x) for x in d["del"])
    (reorder_only if na == nd else substantive).append(f)

print(f"[{lib}] TOTAL: {len(files)}  REORDER-ONLY: {len(reorder_only)}  SUBSTANTIVE: {len(substantive)}")

with open(f"reorder_only_{lib}.txt", "w") as fh:
    fh.write("\n".join(sorted(reorder_only)) + ("\n" if reorder_only else ""))
with open(f"substantive_changes_{lib}.txt", "w") as fh:
    fh.write("\n".join(sorted(substantive)) + ("\n" if substantive else ""))
