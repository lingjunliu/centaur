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

    input_tensor = input_tensor.reshape(1, 1, 1, -1)
    unfold = torch.nn.Unfold(kernel_size=(1,kernel_size), dilation=(dilation,dilation), padding=(padding,padding), stride=(stride,stride))
    result = unfold(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    kernel_size = input_dict.get("kernel_size")
    dilation = input_dict.get("dilation", 1)
    padding = input_dict.get("padding", 0)
    stride = input_dict.get("stride", 1)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.reshape(input_tensor, [1, 1, 1, input_tensor.shape[0]])
        result = tf.image.extract_patches(
            images=input_tensor,
            sizes=[1, 1, 1, kernel_size],
            strides=[1, 1, 1, stride],
            rates=[1, 1, 1, dilation],
            padding='VALID' if padding == 0 else 'SAME'
        )
        result = tf.reshape(result, [1, -1])

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3, 4], dtype=np.float32),
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