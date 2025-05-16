import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
from tensorflow.experimental import dlpack as tf_dlpack
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

def torch_version(input_dict, cpu=True):
    input_tensor = torch.tensor(input_dict["input"])
    if not cpu:
        input_tensor = input_tensor.cuda()
    dlpack_tensor = torch.to_dlpack(input_tensor)
    if not cpu:
        input_tensor = input_tensor.cpu()
    return {"result": dlpack_tensor}

def tensorflow_version(input_dict, cpu=True):
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        dlpack_tensor = tf_dlpack.to_dlpack(input_tensor)
    return {"result": dlpack_tensor}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    torch_dlpack = torch_result['result']
    tf_dlpack = tf_result['result']
    
    torch_array = np.from_dlpack(torch_dlpack)
    tf_array = np.from_dlpack(tf_dlpack)

    assert np.allclose(torch_array, input_data["input"], atol=A_TOL), "Torch DLPack conversion failed"
    assert np.allclose(tf_array, input_data["input"], atol=A_TOL), "TF DLPack conversion failed"
    assert np.allclose(torch_array, tf_array, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()