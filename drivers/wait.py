import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    future = torch.jit.fork(lambda x: x + 1, torch.tensor(input_dict["input"]))

    if not cpu:
        pass
    
    result = torch.jit.wait(future)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    from concurrent.futures import Future

    def fork_mimic(func, *args):
        future = Future()
        try:
            result = func(*args)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        return future

    def wait_mimic(future):
        return future.result()

    def add_one(x):
        return x + 1
    
    future = fork_mimic(add_one, tf.constant(input_dict["input"]))
    
    result = wait_mimic(future)
    
    result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1, 2, 3], dtype=np.int32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()