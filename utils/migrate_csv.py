
import sys

from utils.new_api_utils import get_lib_version

def main():
    csv_file = sys.argv[1]
    lib = sys.argv[2] if len(sys.argv) > 2 else "torch"
    out_str = ''
    with open(csv_file, "r") as f:
        for line in f.readlines():
            tokens = line.strip().split(",")
            out_str += f"{get_lib_version(tokens[0], lib)},{','.join(tokens[1:])}\n"
    
    with open(csv_file, "w") as f:
        f.write(out_str)

if __name__ == "__main__":
    main()