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

    result = torch.nn.functional.bilinear(input1, input2, weight, bias=bias)

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

        input1_shape = input1.shape
        input2_shape = input2.shape
        weight_shape = weight.shape

        batch_size = input1_shape[0]
        hidden_dim = weight_shape[0]

        input1_reshaped = tf.reshape(input1, [batch_size * input1_shape[1], input1_shape[2]])
        input2_reshaped = tf.reshape(input2, [batch_size * input2_shape[1], input2_shape[2]])
        
        w_reshaped = tf.transpose(weight, perm=[1, 2, 0])
        
        output = tf.matmul(input1_reshaped, w_reshaped[:,:,0])
        
        output = tf.matmul(output, tf.transpose(input2_reshaped))
        
        output = tf.reshape(tf.linalg.diag_part(output), (batch_size, input1_shape[1]))

        if bias is not None:
            output = tf.add(output, bias)

        result = output.numpy()
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