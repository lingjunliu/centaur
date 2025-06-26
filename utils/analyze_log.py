import os, sys
from rapidfuzz import fuzz
from utils.misc import create_subdir, get_tmp_dir
from utils.new_api_utils import get_lib_version

def analyze(logfile, threshold=80, expand=False):
    exception_messages = set()

    with open(logfile, 'r') as file:
        lines = file.readlines()
        for line in lines:
            tokens = line.strip().rsplit("|", 2)
            if len(tokens) < 3:
                continue
            levelname = tokens[1].strip()
            if levelname == "ERROR":
                exception_message = tokens[2].strip()
                exception_messages.add(exception_message)
    
    exception_messages = sorted(exception_messages, key=lambda x: x.lower())
    groups = {
        exception_messages[0]: []
    }
    for message in exception_messages:
        assigned = False
        for key in groups:
            if fuzz.ratio(key, message) >= threshold:
                groups[key].append(message)
                assigned = True
                break
        
        if not assigned:
            groups[message] = [message]

    print(f"Grouped exceptions:\n{'-----' * 20}")

    for key, messages in groups.items():
        print(f"{key} ({len(messages)} occurrences)")
        if expand:
            print('\n')
            for msg in messages:
                print(f"  - {msg}")

    print(f"{'-----' * 20}\nTotal unique exceptions found: {len(exception_messages)}")

def main():
    api = sys.argv[1]
    
    lib = "torch"
    api = get_lib_version(api, lib=lib)  # Assuming torch as the default library
    tmp_results = create_subdir(get_tmp_dir(), "rand_results")
    logfile = os.path.join(tmp_results, f"{api}_excp.log")
    THRESHOLD = 80      # Threshold for similarity for exception messages
    EXPAND = False      # Expand similar exception messages
    
    if not os.path.exists(logfile):
        print(f"Log file {logfile} does not exist.")
        return
    
    print(f"[{api}] Analyzing log file: {logfile}\n")
    analyze(logfile, threshold=THRESHOLD, expand=EXPAND)

if __name__ == "__main__":
    main()