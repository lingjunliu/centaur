import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    result = torch.get_num_interop_threads()
    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    import torch

    if cpu:
        num_threads = torch_version(input_dict, cpu=cpu)["result"].item()
    else:
        num_threads = 1
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    num_threads = tf.config.threading.get_inter_op_parallelism_threads()
    return {"result": np.array(num_threads)}

def main():
    A_TOL = 0.01
    input_data = {}

    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"



    torch_result_cuda = torch_version(input_data, cpu=False)
    tf_result_cuda = tensorflow_version(input_data, cpu=False)

    assert np.allclose(torch_result_cuda["result"], tf_result_cuda["result"], atol=A_TOL), "CUDA Results do not match"

    print("Success")

if __name__ == "__main__":
    main()