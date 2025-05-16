import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    p = input.get("p", 2.0)
    dim = input.get("dim", 1)
    eps = input.get("eps", 1e-12)
    out = None  # Since out will always be None as specified

    # Move to GPU if not using CPU
    if not cpu:
        input_tensor = input_tensor.to('cuda')

    # Apply to torch.nn.functional.normalize
    normalized = torch.nn.functional.normalize(input_tensor, p=p, dim=dim, eps=eps, out=out)

    # Move back to CPU for consistency in output
    if not cpu:
        normalized = normalized.cpu()

    return {"normalized": normalized.numpy()}

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
        p = input.get("p", 2.0)
        axis = input.get("dim", 1)
        epsilon = input.get("eps", 1e-12)

        # Apply to TensorFlow equivalent
        norm = tf.norm(input_tensor, ord=p, axis=axis, keepdims=True)
        normalized = input_tensor / tf.maximum(norm, epsilon)

    return {"normalized": normalized.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "p": 2.0,
        "dim": 1,
        "eps": 1e-12
    }

    # Torch example
    torch_result = torch_version(input_data, cpu=True)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data, cpu=True)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_normalized = torch_result["normalized"]
    tf_normalized = tf_result["normalized"]

    if np.allclose(torch_normalized, tf_normalized, atol=1e-7):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()