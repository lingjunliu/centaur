import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn as nn

    input_tensor = torch.tensor(input_dict["input"])
    out_features = input_dict["out_features"]
    bias = input_dict.get("bias", True)

    if not cpu:
        input_tensor = input_tensor.cuda()

    lazy_linear = nn.LazyLinear(out_features, bias=bias)

    if not cpu:
        lazy_linear = lazy_linear.cuda()

    result = lazy_linear(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        out_features = input_dict["out_features"]
        bias_flag = input_dict.get("bias", True)

        in_features = int(input_tensor.shape[-1])

        limit = np.sqrt(1 / in_features)
        kernel_initializer = tf.keras.initializers.RandomUniform(
            minval=-limit, maxval=limit
        )

        kernel = tf.Variable(kernel_initializer(shape=(out_features, in_features)), dtype=tf.float32)
        
        result = tf.matmul(input_tensor, kernel, transpose_b = False)

        if bias_flag:
            bias_initializer = tf.keras.initializers.RandomUniform(
                minval=-limit, maxval=limit
            )
            bias = tf.Variable(bias_initializer(shape=(out_features,)), dtype=tf.float32)
            result = tf.add(result, bias)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "out_features": 4,
        "bias": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()