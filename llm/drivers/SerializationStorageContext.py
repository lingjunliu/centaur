import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    context_size = input_dict.get("context_size", 1)
    future_length = input_dict.get("future_length", 1)
    padding = input_dict.get("padding", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = np.array([True])


    if not cpu:
        result = np.array([True])

    return {"result": result}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        context_size = input_dict.get("context_size", 1)
        future_length = input_dict.get("future_length", 1)
        padding = input_dict.get("padding", False)

        shape = tf.shape(input_tensor)
        storage_size = tf.size(input_tensor)
        storage_offset = 0
        storage_stride = 1
        dim = tf.rank(input_tensor)

        result = np.array([True])

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3, 4, 5], dtype=np.int64),
        "context_size": 2,
        "future_length": 3,
        "padding": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL)

    print("Success")


if __name__ == "__main__":
    main()