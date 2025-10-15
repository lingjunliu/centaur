import sys

def main():
    old = sys.argv[1]
    new = sys.argv[2]

    with open(old) as f:
        old_lines = f.readlines()

    with open(new) as f:
        new_lines = f.readlines()

    old_coverage = {}
    new_coverage = {}

    for line in old_lines:
        try:
            api, coverage, _ = line.strip().split(',')
            old_coverage[api] = float(coverage)
        except ValueError:
            continue

    for line in new_lines:
        try:
            api, coverage, _ = line.strip().split(',')
            new_coverage[api] = float(coverage)
        except ValueError:
            continue

    regressing_apis = []
    for api in old_coverage:
        if api in new_coverage:
            if new_coverage[api] < old_coverage[api]:
                regressing_apis.append((api, old_coverage[api], new_coverage[api]))

    print("API,Old_Coverage,New_Coverage")
    for api, old_cov, new_cov in regressing_apis:
        print(f"{api},{old_cov},{new_cov}")

if __name__ == "__main__":
    main()