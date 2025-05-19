import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    indices_tensor = torch.tensor(input["indices"], dtype=torch.int64)
    kernel_size = input["kernel_size"]
    stride = input.get("stride", kernel_size)
    padding = input.get("padding", 0)
    output_size = input.get("output_size", None)

    if not cpu:
        input_tensor = input_tensor.cuda()
        indices_tensor = indices_tensor.cuda()

    # Apply to torch.nn.functional.max_unpool2d
    output_tensor = torch.nn.functional.max_unpool2d(
        input_tensor, indices_tensor, kernel_size, stride, padding, output_size
    )

    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"max_unpool2d_output": output_tensor.numpy()}

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
        indices_tensor = tf.constant(input["indices"], dtype=tf.int64)
        kernel_size = input["kernel_size"]
        stride = input.get("stride", kernel_size)
        padding = input.get("padding", 0)
        output_size = input.get("output_size", None)

        # Calculate output shape based on the provided parameters
        input_shape = tf.shape(input_tensor, out_type=tf.int64)
        if output_size is None:
            output_height = (input_shape[2] - 1) * stride - 2 * padding + kernel_size
            output_width = (input_shape[3] - 1) * stride - 2 * padding + kernel_size
            output_size = [
                input_shape[0],
                input_shape[1],
                output_height,
                output_width,
            ]
        else:
            output_height, output_width = output_size[2], output_size[3]

        batch_size, channels, height, width = input_shape[0], input_shape[1], output_height, output_width

        # Initialize the output tensor
        output_tensor = tf.zeros(output_size)
        
        for b in range(batch_size):
            for c in range(channels):
                input_flat = tf.reshape(input_tensor[b, c], [-1])
                indices_flat = tf.reshape(indices_tensor[b, c], [-1])

                # Create an index matrix for scatter_nd
                scatter_indices = tf.expand_dims(indices_flat, 1)

                updates = tf.scatter_nd(scatter_indices, input_flat, [height * width])
                updates = tf.reshape(updates, [height, width])

                # Assign the updates to the correct place in output_tensor
                output_tensor = output_tensor + tf.tensor_scatter_nd_update(
                    output_tensor[b:b+1, c:c+1], [[0, 0]], tf.expand_dims(updates, 0)
                )

        return {"max_unpool2d_output": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(1, 1, 2, 2).astype(np.float32),
        "indices": np.array([[[[0, 1], [2, 3]]]], dtype=np.int64),
        "kernel_size": 2,
        "stride": None,
        "padding": 0,
        "output_size": [1, 1, 4, 4]
    }

    # Ensure reproducibility

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results using numpy
    if np.allclose(torch_result["max_unpool2d_output"], tf_result["max_unpool2d_output"], atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()