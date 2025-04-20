import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    size = input.get("size", None)
    scale_factor = input.get("scale_factor", None)

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply to torch.nn.functional.interpolate
    if size is not None:
        output = torch.nn.functional.interpolate(
            input_tensor, size=size, mode='nearest')
    elif scale_factor is not None:
        output = torch.nn.functional.interpolate(
            input_tensor, scale_factor=scale_factor, mode='nearest')
    else:
        raise ValueError("Either size or scale_factor must be provided")

    if not cpu:
        output = output.cpu()

    return {"upsampling_output": output.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        size = input.get("size", None)
        scale_factor = input.get("scale_factor", None)

        # Transpose dimensions from NCHW to NHWC for TensorFlow
        input_tensor = tf.transpose(input_tensor, [0, 2, 3, 1])

        # Apply to TensorFlow equivalent
        input_shape = tf.shape(input_tensor)

        if size is not None:
            output = tf.image.resize(
                input_tensor, size=size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        elif scale_factor is not None:
            new_height = tf.cast(input_shape[1] * scale_factor[0], tf.int32)
            new_width = tf.cast(input_shape[2] * scale_factor[1], tf.int32)
            output = tf.image.resize(
                input_tensor, size=[new_height, new_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        else:
            raise ValueError("Either size or scale_factor must be provided")

        # Transpose dimensions back from NHWC to NCHW for comparison
        output = tf.transpose(output, [0, 3, 1, 2])

        return {"upsampling_output": output.numpy()}

def main():
    # Example input for size variant
    input_data_size = {
        "input": np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2),
        "size": [4, 4],
        "scale_factor": None
    }

    # Example input for scale_factor variant
    input_data_scale = {
        "input": np.arange(1, 5, dtype=np.float32).reshape(1, 1, 2, 2),
        "size": None,
        "scale_factor": (2, 2)  # specify as a tuple for both height and width factors
    }

    # Torch example with size
    torch_result_size = torch_version(input_data_size)
    print("Torch result with size:", torch_result_size)

    # TensorFlow example with size
    tf_result_size = tensorflow_version(input_data_size)
    print("TensorFlow result with size:", tf_result_size)

    if np.allclose(torch_result_size["upsampling_output"], tf_result_size["upsampling_output"], atol=1e-5):
        print("Equal result for size variant")
    else:
        print("Not equal result for size variant")

    # Torch example with scale_factor
    torch_result_scale = torch_version(input_data_scale)
    print("Torch result with scale factor:", torch_result_scale)

    # TensorFlow example with scale_factor
    tf_result_scale = tensorflow_version(input_data_scale)
    print("TensorFlow result with scale factor:", tf_result_scale)

    if np.allclose(torch_result_scale["upsampling_output"], tf_result_scale["upsampling_output"], atol=1e-5):
        print("Equal result for scale factor variant")
    else:
        print("Not equal result for scale factor variant")

if __name__ == "__main__":
    main()