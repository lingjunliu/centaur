import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    kernel_size = input_dict.get("kernel_size")
    dilation = input_dict.get("dilation", 1)
    padding = input_dict.get("padding", 0)
    stride = input_dict.get("stride", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()

    unfold = torch.nn.Unfold(kernel_size=(kernel_size,)*2, dilation=(dilation,)*2, padding=(padding,)*2, stride=(stride,)*2)
    side = int(np.sqrt(input_tensor.shape[0]))
    input_tensor = input_tensor.reshape(1, 1, side, side)
    result = unfold(input_tensor)

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
        kernel_size = input_dict.get("kernel_size")
        dilation = input_dict.get("dilation", 1)
        padding = input_dict.get("padding", 0)
        stride = input_dict.get("stride", 1)

        side = int(np.sqrt(input_tensor.shape[0]))
        input_tensor = tf.reshape(input_tensor, [1, side, side, 1])

        result = tf.nn.conv2d(
            input=input_tensor,
            filters=tf.ones((kernel_size, kernel_size, 1, 1)),
            strides=[1, stride, stride, 1],
            padding='VALID' if padding == 0 else 'SAME',
            dilations=[1, dilation, dilation, 1]
        )

        patches = []
        for i in range(result.shape[1]):
            for j in range(result.shape[2]):
                patch = input_tensor[:, i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size, :]
                patches.append(tf.reshape(patch, [-1]))
        result = tf.stack(patches)

        result = tf.transpose(result)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32),
        "kernel_size": 2,
        "stride": 1,
        "padding": 0,
        "dilation": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()