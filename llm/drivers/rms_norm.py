import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    eps = input_dict.get("eps", 1e-08)
    normalized_shape = input_dict.get("normalized_shape", input_tensor.shape)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()

    result = torch.nn.functional.rms_norm(input_tensor, normalized_shape, weight, eps)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        weight = tf.constant(input_dict["weight"], dtype=tf.float32)
        eps = input_dict.get("eps", 1e-08)
        normalized_shape = input_dict.get("normalized_shape", input_tensor.shape)

        reduction_axes = list(range(len(input_tensor.shape) - len(normalized_shape), len(input_tensor.shape)))
        ms = tf.reduce_mean(tf.square(input_tensor), axis=reduction_axes, keepdims=True)
        rms = tf.sqrt(ms + eps)
        result = input_tensor / rms * weight
        result = result.numpy()
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "weight": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "eps": 1e-05,
        "normalized_shape": (3,)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()