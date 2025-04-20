import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    upscale_factor = input["upscale_factor"]

    # Apply PixelShuffle
    pixel_shuffle = torch.nn.PixelShuffle(upscale_factor)
    output_tensor = pixel_shuffle(input_tensor)

    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"output": output_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        upscale_factor = input["upscale_factor"]

        # Apply equivalent PixelShuffle
        input_shape = tf.shape(input_tensor)
        batch_size, in_depth, in_height, in_width = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        out_depth = in_depth // (upscale_factor * upscale_factor)
        out_height = in_height * upscale_factor
        out_width = in_width * upscale_factor

        # Reshape and transpose
        reshaped = tf.reshape(input_tensor, [batch_size, out_depth, upscale_factor, upscale_factor, in_height, in_width])
        transposed = tf.transpose(reshaped, [0, 1, 4, 2, 5, 3])
        output_tensor = tf.reshape(transposed, [batch_size, out_depth, out_height, out_width])

        return {"output": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(1, 9, 4, 4).astype(np.float32),
        "upscale_factor": 3
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result["output"].shape)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result["output"].shape)

    # Compare the results
    torch_output = torch_result["output"]
    tf_output = tf_result["output"]

    assert np.allclose(torch_output, tf_output), "Outputs are not equal"
    print("Outputs are equal")

if __name__ == "__main__":
    main()