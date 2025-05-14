import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input1 = torch.tensor(input_dict["input1"])
    input2 = torch.tensor(input_dict["input2"])
    weight = torch.tensor(input_dict["weight"])
    bias = input_dict.get("bias", None)
    if bias is not None:
      bias = torch.tensor(bias)

    if not cpu:
        input1 = input1.cuda()
        input2 = input2.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    result = torch.nn.functional.bilinear(input1, input2, weight, bias)

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
        input1 = tf.constant(input_dict["input1"])
        input2 = tf.constant(input_dict["input2"])
        weight = tf.constant(input_dict["weight"])
        bias = input_dict.get("bias", None)
        if bias is not None:
            bias = tf.constant(bias)

        input1_shape = tf.shape(input1)
        input2_shape = tf.shape(input2)
        weight_shape = tf.shape(weight)

        input1_reshaped = tf.reshape(input1, [-1, input1_shape[-1]])
        input2_reshaped = tf.reshape(input2, [-1, input2_shape[-1]])

        w_reshaped = tf.reshape(weight, [weight_shape[0], weight_shape[1] * weight_shape[2]])
        result = tf.matmul(input1_reshaped, tf.transpose(tf.slice(w_reshaped, [0, 0], [weight_shape[0], input_dict["weight"].shape[1]*input_dict["weight"].shape[2]])))

        result = tf.matmul(result, input2_reshaped, transpose_b=True)

        result = tf.reshape(result, tf.concat([tf.shape(input1)[:-1], tf.shape(input2)[:-1], [weight_shape[0]]], axis=0))

        if bias is not None:
            result = tf.add(result, bias)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input1": np.random.rand(2, 3, 4).astype(np.float32),
        "input2": np.random.rand(2, 3, 5).astype(np.float32),
        "weight": np.random.rand(6, 4, 5).astype(np.float32),
        "bias": np.random.rand(6).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()