import numpy as np

def torch_flatten(input, cpu=True):
    import torch
    """Flatten a tensor using PyTorch."""

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    start_dim = input.get("start_dim", 1)
    end_dim = input.get("end_dim", -1)

    # Apply to torch.nn.Flatten
    flatten = torch.nn.Flatten(start_dim=start_dim, end_dim=end_dim)
    output_tensor = flatten(input_tensor)

    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"output": output_tensor.numpy()}

def tensorflow_flatten(input, cpu=True):
    import tensorflow as tf
    """Flatten a tensor using TensorFlow."""

    # Set device string based on `cpu` flag
    device_string = "/cpu:0" if cpu else "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        start_dim = input.get("start_dim", 1)
        end_dim = input.get("end_dim", -1)

        # Calculate the shape for flattening
        input_shape = input_tensor.shape.as_list()
        # end_dim might be negative, so convert it to positive
        if end_dim < 0:
            end_dim += len(input_shape)

        num_elements = np.prod(input_shape[start_dim:end_dim + 1])
        new_shape = input_shape[:start_dim] + [int(num_elements)] + input_shape[end_dim + 1:]
        
        # Flatten tensor in TensorFlow by reshaping
        output_tensor = tf.reshape(input_tensor, new_shape)

        return {"output": output_tensor.numpy()}

def main():
    """Main function to run examples and compare results."""
    # Example input
    input_data = {
        "input": np.random.randn(32, 1, 5, 5).astype(np.float32),
        "start_dim": 1,
        "end_dim": -1
    }

    # Torch example
    torch_result = torch_flatten(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_flatten(input_data)
    print("TensorFlow result:", tf_result)

    # Convert outputs to numpy arrays for comparison if necessary
    torch_output = np.array(torch_result["output"])
    tf_output = np.array(tf_result["output"])

    # Assert that outputs are the same
    if np.allclose(torch_output, tf_output, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()
