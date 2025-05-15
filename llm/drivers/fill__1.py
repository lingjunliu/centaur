import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    value = input_dict["value"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    input_tensor.fill_(value)

    if not cpu:
        input_tensor = input_tensor.cpu()

    return {"result": input_tensor.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        value = input_dict["value"]

        input_shape = tf.shape(input_tensor)
        filled_tensor = tf.fill(input_shape, value)

        result = tf.identity(tf.cast(filled_tensor, input_tensor.dtype))

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "value": 5.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()