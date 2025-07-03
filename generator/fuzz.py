import sys
from utils.misc import create_subdir, get_tmp_dir
from datetime import datetime
import subprocess
import logging
import signal

def run_fuzz(api, duration, n_max, lib, seed, print_details, use_reference):
    log_dir = create_subdir(get_tmp_dir(), "crash_logs")
    logfile = f'{log_dir}/{api}.log'

    logger = logging.getLogger(__name__)
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,                                     # Minimum log level
        format='%(asctime)s - %(levelname)s - %(message)s',     # Log format
        filename=logfile,                                       # Log file path
        filemode="w"                                            # Append/Write mode
    )

    mode = "z3" # default mode, optimizer partially implemented

    # alias
    if lib == "tensorflow":
        lib = "tf"
    elif lib == "pytorch":
        lib = "torch"


    # Logging run config at the beginning
    print(f"Fuzzing with the {api} driver on {lib}. Mode: {mode}, seed: {seed}.\nLog File (crash): {logfile}")
    print('Started fuzzing at', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    if mode.strip().lower() == "z3":
        cmd = ["python3", "-m", "generator.harness_z3", api, str(duration), str(n_max), lib, str(seed), str(print_details).lower(), str(use_reference).lower()]
        try:
            output = subprocess.run(cmd, capture_output=True)
        except Exception as e:
            print(f"ERROR while fuzzing. Faced exception: {str(e)} \nExiting...")
            return

        out, err = output.stdout.decode(), output.stderr.decode()
        print(out)

        if output.returncode > 0:
            print(f"Process terminated with error code: {output.returncode}")
            print(f"Error message: {err}\nExiting...")
        elif output.returncode < 0:
            exception_message = signal.Signals(-output.returncode).name
            print(f"{api} crashed with signal: {exception_message}. Please check the last abstract input in the log file under .tmp/fuzz_logs/{api}.log")
            logger.error(f"{api} crashed with signal: {exception_message}")
            if err:
                logger.info(f"Error message: {err}")
        else:
            print(f"{api} fuzzing completed successfully.")
    else:
        raise Exception(f"Unsupported mode: {mode}. Supported mode: z3.")
    

def main():
    if len(sys.argv) < 3:
        print("Usage: python fuzz.py <api> <duration> <n_max, optional> <lib, default: torch> <seed, optional> <print_details, optional>")
        return
    
    api = sys.argv[1]
    duration = int(sys.argv[2])
    n_max = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    lib = sys.argv[4] if len(sys.argv) > 4 else "torch"
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    print_details = sys.argv[6].lower() == 'true' if len(sys.argv) > 6 else False
    use_reference = sys.argv[7].lower() == 'true' if len(sys.argv) > 7 else False
    
    run_fuzz(api, duration, n_max, lib, seed, print_details, use_reference)

if __name__ == "__main__":
    main()