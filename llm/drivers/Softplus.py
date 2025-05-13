import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    beta = input_dict.get("beta", 1.0)
    threshold = input_dict.get("threshold", 20.0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    softplus = torch.nn.Softplus(beta=beta, threshold=threshold)
    result = softplus(input_tensor)

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
        input_tensor = tf.convert_to_tensor(input_dict["input"], dtype=tf.float32)
        beta = input_dict.get("beta", 1.0)
        threshold = input_dict.get("threshold", 20.0)

        result = tf.where(
            input_tensor > threshold,
            input_tensor,
            (tf.math.log(tf.math.exp(beta * (input_tensor - threshold)) + 1) / beta) + threshold
        )
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([-1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        "beta": 0.5,
        "threshold": 1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()