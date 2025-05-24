import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    apis_successful = set()
    dict_str = ""
    with open(f"{CUR_DIR}/inputs.csv") as f:
        for line in f.readlines():
            tokens = line.strip().split(",")
            if tokens[-1] == "0":
                apis_successful.add(tokens[0])
            else:
                apis_successful = apis_successful - set(tokens[0])

    needs_inputs = []
    with open(f"{CUR_DIR}/needs_inputs.txt", "r") as f:
        for line in f.readlines():
            api = line.strip()
            if api not in apis_successful:
                needs_inputs.append(api)
            else:
                dict_str += f"    '{api}': valid_inputs.{api}_inputs(),\n"
    
    with open(f"{CUR_DIR}/needs_inputs.txt", "w") as f:
        for api in needs_inputs:
            f.write(f"{api}\n")

    print(dict_str)

if __name__ == "__main__":
    main()