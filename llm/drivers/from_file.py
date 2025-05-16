import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

def torch_version(input_dict, cpu=True):

    input_tensor = torch.tensor(input_dict["input"])
    dtype = input_dict.get("dtype", torch.float32)
    layout = input_dict.get("layout", torch.strided)
    requires_grad = input_dict.get("requires_grad", False)
    shared = input_dict.get("shared", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.from_numpy(input_tensor.numpy())
    result = result.to(torch.float32 if dtype == np.float32 else torch.float64)
    result = result.requires_grad_(requires_grad)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        dtype = input_dict.get("dtype", tf.float32)
        requires_grad = input_dict.get("requires_grad", False)

        result = tf.constant(input_tensor.numpy(), dtype=tf.float32 if dtype == np.float32 else tf.float64)

        if requires_grad:
            result = tf.Variable(result)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3], dtype=np.float32),
        "dtype": np.float32,
        "requires_grad": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3], dtype=np.float64),
        "dtype": np.float64,
        "requires_grad": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()