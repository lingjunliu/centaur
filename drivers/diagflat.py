import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    offset = input.get("offset", 0)

    # Apply torch.diagflat
    output_tensor = torch.diagflat(input_tensor, offset=offset)

    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"diagflat": output_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
        
    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        offset = input.get("offset", 0)

        # Flatten the input tensor
        flat_tensor = tf.reshape(input_tensor, [-1])

        # Get the length of the flattened tensor
        flat_length = tf.shape(flat_tensor)[0]

        # Create a square matrix of zeros
        diag_matrix = tf.zeros((flat_length + abs(offset), flat_length + abs(offset)))

        # Create the diagonal matrix
        if offset >= 0:
            indices = [(i, i + offset) for i in range(flat_length)]
        else:
            indices = [(i - offset, i) for i in range(flat_length)]

        indices = np.array(indices)
        diag_matrix = tf.tensor_scatter_nd_update(diag_matrix, indices, flat_tensor)

        return {"diagflat": diag_matrix.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3], [0.2, 0.6]], dtype=np.float32),
        "offset": 1,
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result["diagflat"])

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result["diagflat"])

    # Compare results
    if np.allclose(torch_result["diagflat"], tf_result["diagflat"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()