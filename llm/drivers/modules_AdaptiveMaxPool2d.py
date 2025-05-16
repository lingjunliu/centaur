import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    output_size = input_dict["output_size"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.AdaptiveMaxPool2d(output_size)(input_tensor)

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
        output_size = input_dict["output_size"]

        input_shape = tf.shape(input_tensor)
        batch_size = input_shape[0]
        channels = input_shape[1]
        input_height = input_shape[2]
        input_width = input_shape[3]

        target_height = output_size[0]
        target_width = output_size[1]

        stride_height = tf.cast(tf.floor(tf.cast(input_height, tf.float32) / tf.cast(target_height, tf.float32)), tf.int32)
        stride_width = tf.cast(tf.floor(tf.cast(input_width, tf.float32) / tf.cast(target_width, tf.float32)), tf.int32)

        kernel_height = input_height - (target_height - 1) * stride_height
        kernel_width = input_width - (target_width - 1) * stride_width

        result = tf.nn.max_pool(
            input_tensor,
            ksize=[1, kernel_height, kernel_width, 1],
            strides=[1, stride_height, stride_width, 1],
            padding='VALID'
        )
        
        result = tf.transpose(result, perm=[0, 3, 1, 2])

        result = tf.image.resize(result, [target_height, target_width])

        result = tf.transpose(result, perm=[0, 2, 3, 1])
        result = tf.transpose(result, perm=[0, 2, 0, 1])

        result = result.numpy()
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32),
        "output_size": (16, 16)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()