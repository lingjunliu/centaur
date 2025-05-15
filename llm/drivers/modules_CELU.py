import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    alpha = input_dict.get("alpha", 1.0)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.functional.celu(input_tensor, alpha=alpha)

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
        alpha = input_dict.get("alpha", 1.0)

        result = alpha * (tf.exp(input_tensor / alpha) - 1) * tf.cast(input_tensor < 0, dtype=tf.float32) + input_tensor * tf.cast(input_tensor >= 0, dtype=tf.float32)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32),
        "alpha": 0.5
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()