import os

from translator import write_rules

def main():
    dir = "../references"
    rules_file = "references"
    write_rules(dir, rules_file)

if __name__ == "__main__":
    main()