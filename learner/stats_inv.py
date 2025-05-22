import os, sys
from utils.misc import get_dir_in_root, get_tmp_dir, read_file_in_root

def main():
    lib = sys.argv[1] if len(sys.argv) > 1 else "torch"

    # alias
    if lib == "pytorch":
        lib = "torch"
    elif lib == "tensorflow":
        lib = "tf"

    rule_to_api = {}
    api_to_rule = {}
    inv_dir = get_dir_in_root(f"invariants_{lib}")
    for file in os.listdir(inv_dir):
        file_path = os.path.join(inv_dir, file)
        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                for line in f.readlines():
                    tokens = line.strip().split(",")
                    api, rule = tokens[0], tokens[2]
                    
                    # rule to api mapping
                    if rule not in rule_to_api:
                        rule_to_api[rule] = set()
                    rule_to_api[rule].add(api)
                    
                    # api to rule mapping
                    if api not in api_to_rule:
                        api_to_rule[api] = set()
                    api_to_rule[api].add(rule)
    
    tmp = get_tmp_dir()
    rule_to_api_csv = os.path.join(tmp, f"rule_to_api_{lib}.csv")
    api_to_rule_csv = os.path.join(tmp, f"api_to_rule_{lib}.csv")
    
    with open(rule_to_api_csv, "w") as f:
        for rule, apis in rule_to_api.items():
            f.write(f"{rule},{len(apis)}\n")
            
    with open(api_to_rule_csv, "w") as f:
        for api, rules in api_to_rule.items():
            f.write(f"{api},{len(rules)}\n")
            
    supported_apis = read_file_in_root("apis.txt")
    to_print = True
    for api in supported_apis:
        if api.strip() not in api_to_rule:
            if to_print:
                print("The following apis do not have invariants yet:")
                to_print = False
            print(api.strip())

if __name__ == "__main__":
    main()