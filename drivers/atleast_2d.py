import numpy as np

def torch_version(input, cpu=True):
    import torch
    """Function to apply torch.atleast_2d on the input tensors."""

    # Unpack input dictionary
    tensors = input["tensors"]

    # Move tensors to appropriate device if needed
    if not cpu:
        tensors = [torch.tensor(t).cuda() for t in tensors]
    else:
        tensors = [torch.tensor(t) for t in tensors]

    # Apply torch.atleast_2d
    result_tensors = torch.atleast_2d(*tensors)

    if not cpu:
        result_tensors = [tensor.cpu() for tensor in result_tensors]

    return {"atleast_2d_result": [tensor.numpy() for tensor in result_tensors]}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    """Function to apply the TensorFlow equivalent of torch.atleast_2d on the input tensors."""

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        tensors = [tf.constant(item) for item in input["tensors"]]

        # Apply the TensorFlow equivalent of atleast_2d (expand dims if necessary)
        result_tensors = [tf.expand_dims(tensor, axis=0) if len(tensor.shape) == 1 else tensor for tensor in tensors]

        result_tensors = [tensor.numpy() for tensor in result_tensors]

    return {"atleast_2d_result": result_tensors}

def main():
    # Example input
    input_data = {
        "tensors": [
            np.array([0.5, 0.3, 0.8], dtype=np.float32),
            np.array([[0.2, 0.6, 0.9]], dtype=np.float32)
        ]
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assertion by converting all tensor outputs to numpy arrays for comparison
    torch_output = [np.array(t) for t in torch_result["atleast_2d_result"]]
    tf_output = [np.array(t) for t in tf_result["atleast_2d_result"]]

    assert all(np.array_equal(t, tf_t) for t, tf_t in zip(torch_output, tf_output)), "Results are not equal"
    print("equal")

if __name__ == "__main__":
    main()