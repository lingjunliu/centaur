import os

from translator import write_rules

def main():
    dir = "../references"
    rules_file = "references"
    lib = "torch"
    write_rules(dir, rules_file, lib=lib)

if __name__ == "__main__":
    main()