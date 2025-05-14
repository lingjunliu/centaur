import os

def api_in_file(api, filename):
    """
    Check if the given API is present in the specified file.
    """
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return False
    with open(filename, "r") as f:
        for line in f.readlines():
            if api in line:
                return True
    return False

def main():
    original_drivers = os.listdir("../drivers")
    new_drivers = os.listdir("drivers")
    
    print(f"Original Drivers: {len(original_drivers)}")
    # for driver in original_drivers:
    #     print(driver)
        
    print(f"\nNew Drivers: {len(new_drivers)}")
    # for driver in new_drivers:
    #     print(driver)
    
    with open("supported_apis.txt", "r") as f:
        supported_apis = set(line.strip() for line in f)
    
    supported_torch_apis = []
    with open("supported.csv", "r") as f:
        for line in f.readlines():
            tokens = line.strip().split(",")
            if tokens[0] in supported_apis:
                supported_torch_apis.append(tokens[1])
                
    with open("api_full.txt", "r") as f:
        all_apis = set(line.strip() for line in f)
        all_apis = all_apis.union(supported_torch_apis)
    
    with open("needs_driver.txt", "r") as f:
        needs_apis = set(line.strip() for line in f)
    
    driver_generated = set()
    driver_missed = set()
    for api in all_apis:
        if api in needs_apis:
            continue
        if api in supported_torch_apis:
            continue
        basename = api.split(".")[-1]
        if api_in_file(api, os.path.join("drivers", f"{basename}.py")):
            print(f"Driver for {api} generated.")
            driver_generated.add(api)
        else:
            print(f"Driver for {api} not generated.")
            driver_missed.add(api)
            
    to_write = "API,Status\n"
    not_attempted_text = ""
    tried_but_failed = 0
    previously_existed = 0
    success = 0
    not_attempted = 0
    unknown = 0
    for api in sorted(all_apis):
        if api in supported_torch_apis:
            to_write += f"{api},Driver generation successful\n"
            previously_existed += 1
        elif api in needs_apis:
            to_write += f"{api},Driver generation tried but failed\n"
            tried_but_failed += 1
        elif api in driver_generated:
            to_write += f"{api},Driver generation successful\n"
            success += 1
        elif api in driver_missed:
            to_write += f"{api},Driver generation not attempted\n"
            not_attempted += 1
            not_attempted_text += f"{api}\n"
        else:
            to_write += f"{api},Unknown\n"
            unknown += 1
    
    with open("driver_status.csv", "w") as f:
        f.write(to_write)
        
    with open("not_attempted.txt", "w") as f:
        f.write(not_attempted_text) 

    print(f"\nStats: {success} driver gen succeeded, {tried_but_failed} tried but failed, {previously_existed} previously existed, {not_attempted} not attempted, {unknown} unknown, {success+tried_but_failed+previously_existed+not_attempted+unknown} total")
    
    empty = 0
    for api in supported_torch_apis:
        if api in needs_apis:
            print(f"Needs: {api}")
        if api not in all_apis:
            print(api)
        if api.strip() == "":
            empty += 1
    print(empty)
if __name__ == "__main__":
    main()