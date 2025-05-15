import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    if not cpu:
        torch.set_num_threads(input_dict.get("threads", torch.get_num_threads()))
    else:
        torch.set_num_threads(input_dict.get("threads", torch.get_num_threads()))
    result = torch.get_num_threads()

    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    import os

    num_threads = input_dict.get("threads", os.cpu_count())

    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)

    result = tf.config.threading.get_intra_op_parallelism_threads()

    return {"result": np.array(result)}

def main():
    A_TOL = 0.01

    input_data = {
        "threads": 4
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "threads": 8
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()