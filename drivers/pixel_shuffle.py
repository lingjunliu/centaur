import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    upscale_factor = input["upscale_factor"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply to torch.nn.functional.pixel_shuffle
    output = torch.nn.functional.pixel_shuffle(input_tensor, upscale_factor)

    if not cpu:
        output = output.cpu()

    return {"pixel_shuffle_output": output.numpy()}

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

        # Apply to TensorFlow equivalent
        output = tf.nn.depth_to_space(input_tensor, block_size=upscale_factor)

        return {"pixel_shuffle_output": output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.rand(1, 4, 2, 2).astype(np.float32),  # NCHW format for PyTorch
        "upscale_factor": 2
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    # Adjust the input for TensorFlow to NHWC format
    input_data_tf = {
        "input": np.transpose(input_data["input"], (0, 2, 3, 1)),  # Convert from NCHW to NHWC for TensorFlow
        "upscale_factor": input_data["upscale_factor"]
    }
    tf_result = tensorflow_version(input_data_tf)
    # Convert the output back to NCHW format for comparison
    tf_result_converted = {
        "pixel_shuffle_output": np.transpose(tf_result["pixel_shuffle_output"], (0, 3, 1, 2))
    }
    print("TensorFlow result:", tf_result_converted)

    # Compare outputs
    if np.allclose(torch_result["pixel_shuffle_output"], tf_result_converted["pixel_shuffle_output"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()