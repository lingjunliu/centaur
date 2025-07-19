import sys

def main():
    old = sys.argv[1]
    new = sys.argv[2]
    out = sys.argv[3]

    with open(old, "r") as f:
        old_data = f.readlines()

    with open(new, "r") as f:
        new_data = f.readlines()

    common = set()
    for line in new_data:
        tokens = line.strip().split(",")
        for old_line in old_data:
            old_tokens = old_line.strip().split(",")
            if tokens[0] == old_tokens[0]:
                common.add(tokens[0])
                break

    with open(out, "w") as f:
        for line in new_data:
            f.write(line)
        for line in old_data:
            tokens = line.strip().split(",")
            if tokens[0] not in common:
                f.write(line)

if __name__ == "__main__":
    main()