import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    output_size = input_dict["output_size"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    adaptive_avg_pool3d = torch.nn.AdaptiveAvgPool3d(output_size)
    result = adaptive_avg_pool3d(input_tensor)

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
        output_size = input_dict["output_size"]

        input_shape = input_tensor.shape
        batch_size = input_shape[0]
        channels = input_shape[1]
        depth = input_shape[2]
        height = input_shape[3]
        width = input_shape[4]

        target_depth = output_size[0]
        target_height = output_size[1]
        target_width = output_size[2]

        stride_depth = depth // target_depth
        stride_height = height // target_height
        stride_width = width // target_width

        kernel_depth = depth - (target_depth - 1) * stride_depth
        kernel_height = height - (target_height - 1) * stride_height
        kernel_width = width - (target_width - 1) * stride_width

        result = tf.nn.avg_pool3d(
            input_tensor,
            ksize=[1, kernel_depth, kernel_height, kernel_width, 1],
            strides=[1, stride_depth, stride_height, stride_width, 1],
            padding='VALID'
        )

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 8, 8, 8).astype(np.float32),
        "output_size": (4, 4, 4)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()