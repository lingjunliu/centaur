import sys

def main():
    action = sys.argv[1]    # u: union, i: intersection, d: difference
    file1 = sys.argv[2]
    file2 = sys.argv[3]
    output_file = sys.argv[4]
    split = sys.argv[5].lower().startswith('s')  if len(sys.argv) > 5 else False

    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        set1 = set([line.strip().split(',')[0] for line in f1.readlines()]) if split else set([line.strip() for line in f1.readlines()])
        set2 = set([line.strip().split(',')[0] for line in f2.readlines()]) if split else set([line.strip() for line in f2.readlines()])

    if action == 'u':
        result = set1.union(set2)
    elif action == 'i':
        result = set1.intersection(set2)
    elif action == 'd':
        result = set1.difference(set2)
    else:
        raise ValueError("Invalid action. Use 'u' for union, 'i' for intersection, or 'd' for difference.")

    with open(output_file, 'w') as out_f:
        for item in sorted(result):
            out_f.write(f"{item}\n")

    print(f"[Mode: {action}] Result written to {output_file} ({len(result)} lines)")

if __name__ == "__main__":
    main()