import numpy as np
import tensorflow as tf
import torch

def torch_version(input_dict, cpu=True):
    input_tensor = torch.tensor(input_dict["input"])
    other_tensor = torch.tensor(input_dict["other"]) if isinstance(input_dict["other"], np.ndarray) else input_dict["other"]

    if not cpu:
        input_tensor = input_tensor.cuda()
        if isinstance(other_tensor, torch.Tensor):
            other_tensor = other_tensor.cuda()

    result = torch.fmod(input_tensor, other_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.convert_to_tensor(input_dict["input"], dtype=tf.float32)
        other_tensor = tf.convert_to_tensor(input_dict["other"], dtype=tf.float32) if isinstance(input_dict["other"], np.ndarray) else tf.convert_to_tensor(input_dict["other"], dtype=tf.float32)

        result = tf.raw_ops.FloorMod(x=input_tensor, y=other_tensor)

        result = result.numpy()

        sign_mask = tf.math.sign(input_tensor) * tf.math.sign(other_tensor) < 0
        result = tf.where(sign_mask, result + other_tensor, result)

        abs_mask = tf.math.abs(result) >= tf.math.abs(other_tensor)
        result = tf.where(abs_mask, result - other_tensor, result)

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([-3., -2, -1, 1, 2, 3], dtype=np.float32),
        "other": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([1, 2, 3, 4, 5], dtype=np.float32),
        "other": -1.5
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()