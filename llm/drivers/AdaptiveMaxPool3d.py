import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

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
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    output_size = input_dict["output_size"]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_shape = input_tensor.shape

        def get_divisible_shape(original_shape, target_shape):
            new_shape = []
            for i in range(len(original_shape)):
                new_shape.append(original_shape[i] // target_shape[i] * target_shape[i])
            return tuple(new_shape)

        input_shape_divisible = get_divisible_shape(input_shape[:3], output_size)
        input_tensor_resized = tf.image.resize(input_tensor, input_shape_divisible[:2])
        input_tensor_resized = tf.reshape(input_tensor_resized, (input_shape_divisible[0], input_shape_divisible[1], input_shape_divisible[2], input_shape[3], input_shape[4]))
        result = tf.keras.layers.MaxPool3D(pool_size=(input_shape_divisible[0] // output_size[0], input_shape_divisible[1] // output_size[1], input_shape_divisible[2] // output_size[2]), strides=(input_shape_divisible[0] // output_size[0], input_shape_divisible[1] // output_size[1], input_shape_divisible[2] // output_size[2]))(tf.expand_dims(input_tensor_resized, axis=0))
        result = tf.squeeze(result, axis=0)
        result = result.numpy()
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(8, 9, 10, 5, 6).astype(np.float32),
        "output_size": (2, 3, 2)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()