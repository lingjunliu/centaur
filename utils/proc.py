import multiprocessing, signal, os, psutil, traceback

def get_memory_usage():
    return get_memory_usage_by_pid(os.getpid())

def get_system_memory_usage():
    '''
    Get the system memory usage in percentage.
    '''
    return psutil.virtual_memory().percent

def get_memory_usage_by_pid(pid, include_children=True):
    '''
    Get the memory usage of a process by its PID in megabytes.
    '''
    if not psutil.pid_exists(pid):
        return 0
    
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return 0

    # Get memory information (Resident Set Size in bytes)
    memory_info = process.memory_info()
    rss_bytes = memory_info.rss

    memory_mb = rss_bytes / (1024 ** 2) # Convert bytes to megabytes

    if include_children:
        try:
            for child in process.children(recursive=True):
                memory_mb += get_memory_usage_by_pid(child.pid, include_children=False)
        except Exception as e:
            pass

    return memory_mb

def worker(func, return_dict, *args, **kwargs):
    try:
        result = func(*args, **kwargs)
        return_dict["return_code"] = 0
        return_dict["output"] = result
        return_dict["exception_message"] = ""
        return_dict["traceback"] = ""
    except Exception as e:
        return_dict["return_code"] = 1
        return_dict["output"] = None
        return_dict["exception_message"] = f"{e.__class__.__name__}: {str(e)}"
        return_dict["traceback"] = traceback.format_exc()

def run_with_timeout(func, timeout, *args, **kwargs):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    process = multiprocessing.Process(target=worker, args=(func, return_dict) + args, kwargs=kwargs)
    process.start()
    process.join(timeout)
    
    if process.is_alive():
        process.terminate()
        process.join()
        return 2, None, "Timeout"
    elif process.exitcode < 0:
        return process.exitcode, None, signal.Signals(-process.exitcode).name
    elif process.exitcode > 0:
        return process.exitcode, None, os.strerror(process.exitcode)
    else:        
        return return_dict

def _worker_pipe(func, pipe, *args, **kwargs):
    """Subprocess target: runs func and sends result over pipe (no TF tensor serialization)."""
    try:
        func(*args, **kwargs)
        pipe.send({"return_code": 0, "output": None, "exception_message": "", "traceback": ""})
    except Exception as e:
        pipe.send({"return_code": 1, "output": None,
                   "exception_message": f"{e.__class__.__name__}: {str(e)}",
                   "traceback": traceback.format_exc()})
    finally:
        pipe.close()

# Add run_subprocess to isolate TF SIGABRT/SIGILL crashes in a child process
def run_subprocess(func, *args, **kwargs):
    """Like run() but isolated in a child process to survive TF FATAL aborts (SIGABRT/SIGILL).
    Uses a Pipe instead of Manager to avoid serializing TF tensors."""
    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    p = multiprocessing.Process(target=_worker_pipe, args=(func, child_conn) + args, kwargs=kwargs)
    p.start()
    child_conn.close()
    result = None
    if parent_conn.poll(15):
        try:
            result = parent_conn.recv()
        except Exception:
            pass
    parent_conn.close()
    if p.is_alive():
        p.kill()
    p.join(5)
    if p.is_alive():
        p.terminate()
    if result is None:
        ec = p.exitcode if p.exitcode is not None else -9
        return {"return_code": ec, "output": None,
                "exception_message": f"Process exited with code {ec}", "traceback": ""}
    return result

def run(func, *args, **kwargs):
    return_dict = {}
    worker(func, return_dict, *args, **kwargs)
    return return_dict