import os, sys

from .misc import parse_cancelled_jobs, get_tmp_dir

def main():
    lib = sys.argv[1] if len(sys.argv) > 1 else "torch"
    cancelled_infs, cancelled_modls = parse_cancelled_jobs(lib)
    # Reset the log file
    cancelled_jobs_file = os.path.join(get_tmp_dir(), "cancelled_jobs.log")
    backed_up_file = cancelled_jobs_file + ".bak"
    if os.path.exists(backed_up_file):
        os.remove(backed_up_file)
    if os.path.exists(cancelled_jobs_file):
        os.rename(cancelled_jobs_file, cancelled_jobs_file + ".bak")
    
    if len(cancelled_infs) > 0:
        print(f"# of cancelled inference jobs: {len(cancelled_infs)}")
    if len(cancelled_modls) > 0:
        print(f"# of cancelled model jobs: {len(cancelled_modls)}")

if __name__ == "__main__":
    main()