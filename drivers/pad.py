import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack the input dictionary
    input_tensor = torch.tensor(input["input"])
    pad = tuple(input.get("pad", (0, 0)))
    mode = input.get("mode", 'constant')
    value = input.get("value", 0)

    # Apply to torch.nn.functional.pad
    output_tensor = torch.nn.functional.pad(input_tensor, pad, mode, value)

    if not cpu and torch.cuda.is_available():
        output_tensor = output_tensor.cpu()

    return {"padded_tensor": output_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack the input dictionary
        input_tensor = tf.constant(input["input"])
        pad = input.get("pad", (0, 0))
        mode = input.get("mode", 'constant')
        value = input.get("value", 0)

        # Formulating the padding matrix in TensorFlow compatible way
        padding_matrix = [[0, 0] for _ in range(len(input_tensor.shape))]
        pad_len = len(pad) // 2
        for i in range(pad_len):
            padding_matrix[-(i + 1)] = [pad[2 * i], pad[2 * i + 1]]

        if mode == 'constant':
            output_tensor = tf.pad(input_tensor, padding_matrix, mode='CONSTANT', constant_values=value)
        elif mode == 'reflect':
            output_tensor = tf.pad(input_tensor, padding_matrix, mode='REFLECT')
        elif mode == 'replicate':
            output_tensor = tf.pad(input_tensor, padding_matrix, mode='SYMMETRIC')
        elif mode == 'circular':
            # TensorFlow does not natively support circular padding,
            # this requires a custom implementation depending on the context
            raise NotImplementedError("TensorFlow does not support circular padding natively")
        else:
            raise ValueError(f"Unsupported padding mode: {mode}")

    return {"padded_tensor": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(3, 3, 4, 2).astype(np.float32),  # Example 4D tensor as input
        "pad": (1, 1),  # Pad last dimension by 1 on each side
        "mode": 'constant',
        "value": 0
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Check if results are equal
    np.testing.assert_allclose(torch_result["padded_tensor"], tf_result["padded_tensor"], rtol=1e-5)
    print("equal")

if __name__ == "__main__":
    main()