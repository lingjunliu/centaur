import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    output_size = input_dict["output_size"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.AdaptiveMaxPool3d(output_size)(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict["input"])
    output_size = input_dict["output_size"]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_shape = tf.shape(input_tensor)
        batch_size = input_shape[0]
        depth = input_shape[1]
        height = input_shape[2]
        width = input_shape[3]
        channels = input_shape[4]

        result = tf.transpose(input_tensor, perm=[0, 2, 3, 1, 4])
        result = tf.reshape(result, [batch_size * height, width, depth, channels])

        pooled = tf.nn.max_pool(
            result,
            ksize=[1,
                   int(width / output_size[2]) if width > output_size[2] else width,
                   int(depth / output_size[0]) if depth > output_size[0] else depth,
                   1],
            strides=[1,
                     int(width / output_size[2]) if width > output_size[2] else 1,
                     int(depth / output_size[0]) if depth > output_size[0] else 1,
                     1],
            padding='VALID'
        )

        result = tf.image.resize(pooled, size=[output_size[2], output_size[0]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        result = tf.reshape(result, [batch_size, height, output_size[2], output_size[0], channels])
        result = tf.transpose(result, perm=[0, 3, 1, 2, 4])

        s = tf.shape(result)

        result = tf.nn.max_pool3d(
            tf.cast(tf.reshape(result, [1, s[1], s[2], s[3], s[4] * s[0]]), dtype = tf.float32),
            ksize=[1, int(input_shape[2] / output_size[1]) if input_shape[2] > output_size[1] else input_shape[2], 1, 1, 1],
            strides=[1, int(input_shape[2] / output_size[1]) if input_shape[2] > output_size[1] else 1, 1, 1, 1],
            padding="VALID")
        result = tf.reshape(result, [output_size[0], output_size[1], output_size[2], input_shape[4]])
        result = tf.transpose(result, perm=[1, 0, 2, 3])
        result = tf.expand_dims(result, axis = 0)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.2

    input_data = {
        "input": np.random.rand(2, 10, 12, 14, 3).astype(np.float32),
        "output_size": (5, 6, 7)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()