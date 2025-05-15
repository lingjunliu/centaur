import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    p = input_dict.get("p", 2.0)
    dim = input_dict.get("dim", None)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.functional.normalize(input_tensor, p=p, dim=dim)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        p = input_dict.get("p", 2.0)
        dim = input_dict.get("dim", None)

        if dim is None:
            norm = tf.norm(input_tensor, ord=p)
            result = input_tensor / norm
        else:
            norm = tf.norm(input_tensor, ord=p, axis=dim, keepdims=True)
            result = input_tensor / norm

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "p": 2.0,
        "dim": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()