import os
from utils.misc import get_dir_in_root, get_tmp_dir

def main():
    rule_to_api = {}
    api_to_rule = {}
    inv_dir = get_dir_in_root("invariants")
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
    rule_to_api_csv = os.path.join(tmp, "rule_to_api.csv")
    api_to_rule_csv = os.path.join(tmp, "api_to_rule.csv")
    
    with open(rule_to_api_csv, "w") as f:
        for rule, apis in rule_to_api.items():
            f.write(f"{rule},{len(apis)}\n")
            
    with open(api_to_rule_csv, "w") as f:
        for api, rules in api_to_rule.items():
            f.write(f"{api},{len(rules)}\n")

if __name__ == "__main__":
    main()