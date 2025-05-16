import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    num_threads = input_dict.get("num_threads", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    torch.set_num_threads(num_threads)
    result = torch.get_num_threads()

    if not cpu:
        result = torch.tensor(result).cpu().item()
    
    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    num_threads = input_dict.get("num_threads", 1)
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    
    # There isn't a direct equivalent to get the number of threads after setting
    # We'll just return the value that was set
    result = num_threads
    
    return {"result": np.array(result)}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "num_threads": 4
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()