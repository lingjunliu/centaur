import os, sys, pickle, json

def get_signature(api, lib="torch", suffix=0):
    if suffix > 0:
        api = f"{api}_{suffix}"
    
    signatures = original_signatures
    if api not in signatures:
        raise Exception(f"No signature found for {api}")
    
    api_sig = signatures[api]
    simple_sig = {}
    
    # args
    for arg, domain in api_sig["args"].items():
        simple_sig[arg] = domain

    # kwargs
    for arg, domain in api_sig["kwargs"].items():
        simple_sig[arg] = domain

    # inner if available
    if len(api_sig["inner"].keys()) > 0:
        # args
        for arg, domain in api_sig["inner"]["args"].items():
            simple_sig[arg] = domain

        # kwargs
        for arg, domain in api_sig["inner"]["kwargs"].items():
            simple_sig[arg] = domain

    return simple_sig

def main():
    dir = sys.argv[1] if len(sys.argv) > 1 else ".tmp"
    lib = sys.argv[2] if len(sys.argv) > 2 else "torch"
    duration = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    interval = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    out_dir = sys.argv[5] if len(sys.argv) > 5 else ".tmp/centaur"

    os.makedirs(out_dir, exist_ok=True)
    
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(cur_dir, "template.py"), "r") as f:
        template_code = f.read()
    with open(os.path.join(cur_dir, "../signatures.json"), "r") as f:
        signatures = json.load(f)

    subdirs = []
    for i in range(0, duration, interval):
        start_time = i
        end_time = i + interval
        subdir = os.path.join(out_dir, f"{start_time}-{end_time}")
        os.makedirs(subdir, exist_ok=True)
        subdirs.append((start_time, end_time, subdir))

    with open(f"{lib}_apis.txt", "r") as f:
        apis = [line.strip() for line in f.readlines()]
    
    input_dir = os.path.join(dir, "fuzz_inputs")
    cnt = 0
    for api in apis:
        file_name = f"{api}_{lib}_inputs.pkl"
        file_path = os.path.join(input_dir, file_name)
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist, skipping.")
            continue

        with open(file_path, "rb") as f:
            inputs = pickle.load(f)

        base_timestamp = float(inputs[0][0])
        split_inputs = {subdir: [] for _, _, subdir in subdirs}

        for input_entry in inputs:
            timestamp = float(input_entry[0]) - base_timestamp
            for start_time, end_time, subdir in subdirs:
                if end_time == duration and start_time <= timestamp:
                    split_inputs[subdir].append(input_entry)
                    break
                elif start_time <= timestamp < end_time:
                    split_inputs[subdir].append(input_entry)
                    break

        cur_signatures = {}
        for key, sig in signatures.items():
            if key.startswith(api):
                cur_signatures[key] = sig
        for _, _, subdir in subdirs:
            subdir_subdir = os.path.join(subdir, api)
            os.makedirs(subdir_subdir, exist_ok=True)
            with open(os.path.join(subdir_subdir, "driver.py"), "w") as f:
                f.write(template_code.replace("<api>", api))
            with open(os.path.join(subdir_subdir, "signatures.json"), "w") as f:
                json.dump(cur_signatures, f, indent=4)
            out_file = os.path.join(subdir_subdir, file_name)
            with open(out_file, "wb") as f:
                pickle.dump(split_inputs[subdir], f)

        cnt += 1
        print(f"Done with {cnt}/{len(apis)}", end="\r", flush=True)        
    

if __name__ == "__main__":
    main()