import os, shutil, json
from llm.torch_signatures import signatures

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    finalized_drivers_dir = f"{CUR_DIR}/../drivers"
    os.makedirs(finalized_drivers_dir, exist_ok=True)
    finalized_signatures = f"{CUR_DIR}/finalized_signatures.json"
    
    with open(f"{CUR_DIR}/../apis.txt", "r") as f:
        existing_drivers = [driver.strip() for driver in f.readlines()]
    
    with open(f"{CUR_DIR}/new_drivers.txt", "r") as f:
        new_drivers = [driver.strip() for driver in f.readlines()]
    
    driver_to_api = {}
    api_to_driver = {}
    with open(f"{CUR_DIR}/drivers_to_api.csv", "r") as f:
        for line in f.readlines():
            driver, torch_api = line.strip().split(',')
            if driver == "Driver":
                continue    # header
            driver_to_api[driver] = torch_api
            api_to_driver[torch_api] = driver
    
    with open(f"{CUR_DIR}/needs_inputs.txt", "r") as f:
        needs_inputs = set([d.strip() for d in f.readlines()])
    
    sigs = {}
    for driver in set(new_drivers):
        if driver in existing_drivers:
            print(f"Driver {driver} already exists")
        elif driver in driver_to_api:
            torch_api = driver_to_api[driver]
            if torch_api in signatures:                
                driver_path = f"{CUR_DIR}/drivers/{driver}.py"
                if os.path.isfile(driver_path):
                    # We have the drivers and signature
                    needs_inputs.add(driver)
                    # Copying the driver
                    shutil.copy(driver_path, f"{finalized_drivers_dir}/{driver}.py")
                    # Adding signature
                    sigs[driver] = signatures[torch_api]
                else:
                    print(f"Driver {driver}.py not found")
            else:
                print(f"Signature for {driver} ({torch_api}) not found")
        else:
            print(f"No mapping found for {driver}")
    
    with open(f"{CUR_DIR}/needs_inputs.txt", "w") as f:
        f.write('\n'.join(list(needs_inputs)))

    with open(f"{CUR_DIR}/finalized_drivers.txt", "w") as f:
        f.write('\n'.join(list(sigs.keys())))
    
    with open(f"{CUR_DIR}/../apis.txt", "a") as f:
        f.write('\n'.join(list(sigs.keys())))
    
    with open(finalized_signatures, "w") as f:
        json.dump(sigs, f)
        
    print(f"Finalized {len(sigs.keys())} drivers successfully.")

if __name__ == "__main__":
    main()