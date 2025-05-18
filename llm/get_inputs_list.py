import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    apis_successfull = set()
    dict_str = ""
    with open(f"{CUR_DIR}/inputs.csv") as f:
        for line in f.readlines():
            tokens = line.strip().split(",")
            if tokens[-1] == "0":
                apis_successfull.add(tokens[0])
                dict_str += f"    '{tokens[0]}': {tokens[0]}_inputs,\n"

    needs_inputs = []
    with open(f"{CUR_DIR}/needs_inputs.txt", "r") as f:
        for line in f.readlines():
            api = line.strip()
            if api not in apis_successfull:
                needs_inputs.append(api)
    
    with open(f"{CUR_DIR}/needs_inputs.txt", "w") as f:
        for api in needs_inputs:
            f.write(f"{api}\n")

    print(dict_str)

if __name__ == "__main__":
    main()