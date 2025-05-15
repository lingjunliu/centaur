import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict.get("weight", np.ones(input_tensor.shape[-1])), dtype=input_tensor.dtype)
    eps = input_dict.get("eps", 1e-05)
    normalized_shape = input_dict["normalized_shape"]

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()

    module = torch.nn.modules.RMSNorm(normalized_shape, eps=eps, elementwise_affine=True)
    module.weight = torch.nn.Parameter(weight)
    module.bias = None

    result = module(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        weight = tf.constant(input_dict.get("weight", np.ones(input_tensor.shape[-1])), dtype=tf.float32)
        eps = input_dict.get("eps", 1e-05)
        normalized_shape = input_dict["normalized_shape"]

        input_dtype = input_tensor.dtype
        input_tensor = tf.cast(input_tensor, tf.float32)
        weight = tf.cast(weight, tf.float32)

        ms = tf.reduce_mean(tf.square(input_tensor), axis=-1, keepdims=True)
        rms = tf.math.rsqrt(ms + eps)
        output = input_tensor * rms * weight

        output = tf.cast(output, input_dtype)
        result = output.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.0202, 1.0985, 1.3506, -0.6056], [0.0202, 1.0985, 1.3506, -0.6056]], dtype=np.float32),
        "normalized_shape": 4,
        "weight": np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()