import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    kernel_size = input_dict["kernel_size"]
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    output_size = input_dict["output_size"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.functional.fold(input_tensor, output_size=output_size, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

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
        kernel_size = input_dict["kernel_size"]
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        output_size = input_dict["output_size"]

        input_shape = input_tensor.shape
        input_channels = input_shape[1]
        kernel_height, kernel_width = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        output_height, output_width = (output_size, output_size) if isinstance(output_size, int) else output_size
        stride_height, stride_width = (stride, stride) if isinstance(stride, int) else stride
        padding_height, padding_width = (padding, padding) if isinstance(padding, int) else padding
        dilation_height, dilation_width = (dilation, dilation) if isinstance(dilation, int) else dilation
        
        input_tensor = tf.transpose(input_tensor, perm=[0, 2, 1])
        
        patches = tf.image.extract_patches(
            images=tf.reshape(input_tensor, (input_shape[0], input_shape[2], input_shape[1], 1)),
            sizes=[1, kernel_height, kernel_width, 1],
            strides=[1, stride_height, stride_width, 1],
            rates=[1, dilation_height, dilation_width, 1],
            padding='VALID'
        )

        patches_shape = patches.shape
        patches = tf.reshape(patches, (patches_shape[0], patches_shape[1], patches_shape[2], input_channels, kernel_height, kernel_width))

        output = tf.zeros((input_shape[0], output_height, output_width, input_channels))

        for i in range(int(patches_shape[1])):
            for j in range(int(patches_shape[2])):
                patch = patches[:, i, j, :, :, :]
                output_updates = tf.transpose(patch, perm=[0, 1, 2, 3])
                indices = tf.constant([[0, i, j, 0]])
                updates = tf.reshape(output_updates, [input_shape[0], 1, 1, input_channels])

                output = tf.tensor_scatter_nd_add(output, [[k, i, j, l] for k in range(input_shape[0]) for l in range(input_channels)], tf.reshape(output_updates, [-1]))
        output = tf.transpose(output, perm=[0, 3, 1, 2])

        result = output.numpy()
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 4, 9).astype(np.float32),
        "output_size": 4,
        "kernel_size": 2,
        "stride": 1
    }

    input_data["input"] = np.random.rand(1, 4, 9).astype(np.float32)
    torch_result = torch_version(input_data)

    input_data["input"] = np.random.rand(1, 4, 9).astype(np.float32)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()