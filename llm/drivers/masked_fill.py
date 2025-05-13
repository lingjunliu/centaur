import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    mask = torch.tensor(input_dict["mask"])
    value = input_dict.get("value", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        mask = mask.cuda()

    result = input_tensor.masked_fill(mask, value)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        mask = tf.constant(input_dict["mask"], dtype=tf.bool)
        value = input_dict.get("value", 0)

        result = tf.where(mask, tf.cast(value, input_tensor.dtype), input_tensor)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "mask": np.array([True, False, False, True], dtype=bool),
        "value": 5.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "mask": np.array([[True, False], [False, True]], dtype=bool),
        "value": -1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3, 4], dtype=np.int32),
        "mask": np.array([True, False, False, True], dtype=bool),
        "value": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()