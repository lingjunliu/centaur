import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = None
    if "bias" in input_dict:
      bias = torch.tensor(input_dict["bias"])
    pad = input_dict.get("pad", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    result = torch.conv_tbc(input_tensor, weight, bias, pad)

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
        input_tensor = tf.constant(input_dict["input"])
        weight = tf.constant(input_dict["weight"])
        bias = None
        if "bias" in input_dict:
          bias = tf.constant(input_dict["bias"])
        pad = input_dict.get("pad", 0)

        input_shape = input_tensor.shape
        weight_shape = weight.shape

        time_steps = input_shape[0]
        batch_size = input_shape[1]
        input_channels = input_shape[2]

        kernel_size = weight_shape[0]
        output_channels = weight_shape[2]

        padding = 'VALID' if pad == 0 else 'SAME'

        input_tensor_reshaped = tf.transpose(input_tensor, perm=[1, 0, 2])
        input_tensor_reshaped = tf.expand_dims(input_tensor_reshaped, axis=2)

        weight_reshaped = tf.transpose(weight, perm=[0, 2, 1])
        weight_reshaped = tf.expand_dims(weight_reshaped, axis=0)

        conv_result = tf.nn.conv2d(input_tensor_reshaped, weight_reshaped, strides=1, padding=padding)


        conv_result = tf.squeeze(conv_result, axis=2)
        conv_result = tf.transpose(conv_result, perm=[0, 1, 2])

        if bias is not None:
           conv_result = conv_result + bias

        conv_result = tf.transpose(conv_result, perm=[1, 0, 2])

        result = conv_result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_channels = 4
    output_channels = 5
    time_steps = 5
    batch_size = 3
    kernel_size = 3

    input_data = {
        "input": np.random.rand(time_steps, batch_size, input_channels).astype(np.float32),
        "weight": np.random.rand(kernel_size, input_channels, output_channels).astype(np.float32),
        "bias": np.random.rand(output_channels).astype(np.float32),
        "pad": kernel_size - 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()