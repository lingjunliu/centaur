import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    num_threads = input_dict["num_threads"]

    torch.set_num_interop_threads(num_threads)
    result = torch.get_num_interop_threads()

    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    num_threads = input_dict["num_threads"]
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    result = tf.config.threading.get_inter_op_parallelism_threads()

    return {"result": np.array(result)}

def main():
    A_TOL = 0.01
    input_data = {
        "num_threads": 4
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()