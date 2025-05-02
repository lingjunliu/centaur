import multiprocessing, signal, os

def worker(func, return_dict, *args, **kwargs):
    try:
        result = func(*args, **kwargs)
        return_dict["return_code"] = 0
        return_dict["outputs"] = result
        return_dict["exception_message"] = ""
    except Exception as e:
        return_dict["return_code"] = 1
        return_dict["outputs"] = None
        return_dict["exception_message"] = f"{e.__class__.__name__}: {str(e)}"

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
        return return_dict["return_code"], return_dict["outputs"], return_dict["exception_message"]
