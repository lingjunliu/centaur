import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    dim = input.get("dim", None)
    correction = 1
    if "unbiased" in input.keys():
        unbiased = input["unbiased"]
        if not unbiased:
            correction = 0
    elif "correction" in input.keys():
        correction = input["correction"]

    keepdim = input.get("keepdim", False)

    if dim is not None:
        var, mean = torch.var_mean(input_tensor, dim=dim, correction=correction, keepdim=keepdim)
    else:
        var, mean = torch.var_mean(input_tensor, correction=correction)

    if not cpu:
        var, mean = var.cpu(), mean.cpu()

    return {
        "var": var.numpy() if cpu else var.detach().numpy(), 
        "mean": mean.numpy() if cpu else mean.detach().numpy()
    }

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        dim = input.get("dim", None)
        correction = 1
        if "unbiased" in input.keys():
            unbiased = input["unbiased"]
            correction = 1 if unbiased else 0
        elif "correction" in input.keys():
            correction = input["correction"]
        keepdim = input.get("keepdim", False)

        if dim is None:
            mean = tf.reduce_mean(input_tensor)
            var = tf.reduce_sum(tf.square(input_tensor - mean)) / (tf.size(input_tensor, out_type=tf.float32) - correction)
        else:
            mean = tf.reduce_mean(input_tensor, axis=dim, keepdims=keepdim)
            var = tf.reduce_sum(tf.square(input_tensor - mean), axis=dim, keepdims=keepdim) / (
                tf.cast(tf.shape(input_tensor)[dim], tf.float32) - (1.0 if unbiased else 0.0)
            )

        return {
            "var": var.numpy(), 
            "mean": mean.numpy()
        }

def main():
    # Example input for first variant (with dim, unbiased, keepdim)
    input_data_1 = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "dim": 1,
        "unbiased": True,
        "keepdim": True
    }

    # Example input for second variant (only unbiased)
    input_data_2 = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "unbiased": False
    }

    # Test first variant with PyTorch and TensorFlow
    torch_result_1 = torch_version(input_data_1)
    tf_result_1 = tensorflow_version(input_data_1)
    print("Torch result (variant 1):", torch_result_1)
    print("TensorFlow result (variant 1):", tf_result_1)
    assert np.allclose(torch_result_1["var"], tf_result_1["var"]) and np.allclose(torch_result_1["mean"], tf_result_1["mean"]), "not equal"
    print("equal" if np.allclose(torch_result_1["var"], tf_result_1["var"]) and np.allclose(torch_result_1["mean"], tf_result_1["mean"]) else "not equal")

    # Test second variant with PyTorch and TensorFlow
    torch_result_2 = torch_version(input_data_2)
    tf_result_2 = tensorflow_version(input_data_2)
    print("Torch result (variant 2):", torch_result_2)
    print("TensorFlow result (variant 2):", tf_result_2)
    assert np.allclose(torch_result_2["var"], tf_result_2["var"]) and np.allclose(torch_result_2["mean"], tf_result_2["mean"]), "not equal"
    print("equal" if np.allclose(torch_result_2["var"], tf_result_2["var"]) and np.allclose(torch_result_2["mean"], tf_result_2["mean"]) else "not equal")

if __name__ == "__main__":
    main()