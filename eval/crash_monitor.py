import sys, subprocess, signal
from eval.oracle import save_state_oracle, retrieve_state_oracle
from utils.misc import create_subdir, get_tmp_dir

def main():
    api = sys.argv[1]
    lib = sys.argv[2] if len(sys.argv) > 2 else "torch"
    # Optional: low and high values for input generation [low, high)
    low = int(sys.argv[3]) if len(sys.argv) > 3 else -1
    high = int(sys.argv[4]) if len(sys.argv) > 4 else -1

    # alias
    if lib == "pytorch":
        lib = "torch"
    elif lib == "tensorflow":
        lib = "tf"

    cmd = ["python3", "-m", "eval.oracle", api, lib, str(low), str(high)]
    try:
        output = subprocess.run(cmd, capture_output=True)
    except Exception as e:
        print(f"ERROR while running the oracle. Faced exception: {str(e)} \nExiting...")
        return

    out, err = output.stdout.decode(), output.stderr.decode()

    if output.returncode > 0:
        print(f"Process terminated with error code: {output.returncode}")
        print(f"Error message: {err}\nExiting...")
        return
    
    results_dir = create_subdir(get_tmp_dir(), f"oracle_results_{lib}")
    logfile = f'{results_dir}/{api}.out'

    while output.returncode < 0:
        oracle_results = []
        result_summary = {
            "nominal": 0,
            "invalid": 0,
            "cpu_crash": 0,
            "gpu_crash": 0,
            "cpu_excp": 0,
            "gpu_excp": 0,
            "cpu_only_excp": 0,
            "gpu_only_excp": 0,
            "inconsistent": 0,
            "max_diff": 0
        }
        exception_message = signal.Signals(-output.returncode).name
        oracle_results, result_summary = retrieve_state_oracle(api, result_summary, oracle_results, lib=lib)
        # Determine where the crash occurred

        crash_category = None
        with open(logfile, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "CPU execution started" in line.strip():
                    crash_category = "cpu_crash"
                elif "GPU execution started" in line.strip():
                    crash_category = "gpu_crash"
        
        # Updating result
        if crash_category:
            result_summary[crash_category] += 1
        
        low = sum(list(result_summary.values())[:-1])   # excluding max_diff
        oracle_results.append((crash_category, exception_message))
        print(f"{crash_category} | {exception_message}")

        # Save the state
        save_state_oracle(api, result_summary, oracle_results, lib=lib)

        print(f"Input {low-1} raised signal: {exception_message}\nRestarting the oracle...")

        cmd = ["python3", "-m", "eval.oracle", api, lib, str(low), str(high), "resume"]
        try:
            output = subprocess.run(cmd, capture_output=True)
        except Exception as e:
            print(f"ERROR while re-running the oracle. Faced exception: {str(e)} \nExiting...")
            return


if __name__ == "__main__":
    main()