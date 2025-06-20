import os, sys, subprocess
from utils.proc import get_memory_usage_by_pid

def get_job_list(cmd, job_name=None):
    try:
        output = subprocess.run(cmd.split(), capture_output=True).stdout.decode()
        lines = []
        for line in output.splitlines():
            if job_name is None or job_name in line:
                lines.append(line.strip())
        return lines
    except subprocess.CalledProcessError as e:
        print(f"Error executing command '{cmd}': {e}")
        return []

def map_name_id(jobs):
    job_dict = {}
    for job_line in jobs:
        tokens = job_line.split()
        if len(tokens) >= 3:
            job_id = tokens[0]
            job_name = tokens[2]
            job_dict[job_name] = job_id
    
    return job_dict

def map_jobid_pid(jobs):
    pid_jobid_dict = {}
    for job_line in jobs:
        tokens = job_line.split()
        if len(tokens) >= 2:
            pid = tokens[0]
            jobid = tokens[1]
            pid_jobid_dict[jobid] = pid
    
    return pid_jobid_dict

def cancel_slurm_job(job_id):
    try:
        subprocess.run(['scancel', job_id], check=True)
        print(f"Cancelled job {job_id}")
    except subprocess.CalledProcessError as e:
        print(f"Error cancelling job {job_id}: {e}")
        
    

def main():
    if len(sys.argv) < 5:
        print("Usage: python monitor_mem.py <jobname> <elapsed> <spawned> <total>")
        return
    
    job_name = sys.argv[1]
    elapsed = int(sys.argv[2])
    spawned = int(sys.argv[3])
    total = int(sys.argv[4])
    threshold = int(sys.argv[5]) if len(sys.argv) > 5 else 4096  # Memory threshold in MB
    
    persist_period = 60  # persist the printed output every 60 seconds
    
    user = os.getenv('USER')
    
    enqueued_cmd = f'squeue -h --user={user}'
    pending_cmd = f'squeue -h --user={user} --state=PENDING'
    running_cmd = f'squeue -h --user={user} --state=RUNNING'
    pids_cmd = 'scontrol listpids'
    
    enqueued_jobs = get_job_list(enqueued_cmd, job_name)
    pending_jobs = get_job_list(pending_cmd)
    running_jobs = get_job_list(running_cmd)
    pids = get_job_list(pids_cmd)
    
    jobname_jobid = map_name_id(running_jobs)
    jobid_pid = map_jobid_pid(pids)
    
    max_memory = 0
    max_mem_job_name = "default_jobname"
    for job_name, job_id in jobname_jobid.items():
        if job_id in jobid_pid:
            pid = jobid_pid[job_id]
            if not pid.isdigit():
                continue
            mem_usage = get_memory_usage_by_pid(int(pid))
            if mem_usage < 0:
                continue
            if mem_usage > threshold:
                print(f"Job {job_name} (ID: {job_id}, PID: {pid}) is using {mem_usage:.2f} MB memory, exceeding the threshold of {threshold} MB.")
                cancel_slurm_job(job_id)
            else:
                max_memory = max(max_memory, mem_usage)
                max_mem_job_name = job_name if mem_usage == max_memory else max_mem_job_name
    
    message = f"{len(enqueued_jobs)} in queue | {len(running_jobs)} running | {len(pending_jobs)} waiting | {elapsed} seconds elapsed | {spawned}/{total} jobs spawned | Max memory usage: {max_memory:.8f} MB (Job: {max_mem_job_name})"
    
    if elapsed % persist_period == 0:
        print(message)
    else:
        print(message, end='\r', flush=True)

if __name__ == "__main__":
    main()