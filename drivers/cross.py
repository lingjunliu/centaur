import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["other"])
    dim = input.get("dim", None)

    if not cpu:
        input_tensor = input_tensor.cuda()
        other_tensor = other_tensor.cuda()

    # Apply to torch.cross
    result = torch.cross(input_tensor, other_tensor, dim=dim)

    if not cpu:
        result = result.cpu()

    return {"cross_product": result.numpy()}

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
        other_tensor = tf.constant(input["other"])
        dim = input.get("dim", -1)  # Default to the last dimension if not provided

        # Apply TensorFlow equivalent (manual cross product implementation)
        a1, a2, a3 = tf.unstack(input_tensor, axis=dim)
        b1, b2, b3 = tf.unstack(other_tensor, axis=dim)
        cross_product = tf.stack([
            a2 * b3 - a3 * b2,
            a3 * b1 - a1 * b3,
            a1 * b2 - a2 * b1
        ], axis=dim)

        return {"cross_product": cross_product.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "other": np.array([[0.1, 0.4, 0.7], [0.5, 0.2, 0.3]], dtype=np.float32),
        "dim": 1  # Perform cross product along dimension 1
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    torch_cross_product = torch_result["cross_product"]
    tf_cross_product = tf_result["cross_product"]

    if np.allclose(torch_cross_product, tf_cross_product):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()