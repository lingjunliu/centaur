import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    dim = input.get("dim", None)
    dtype = getattr(torch, input.get("dtype", None)) if input.get("dtype", None) else None
    
    # Apply to torch.nn.functional.softmax
    if dtype:
        input_tensor = input_tensor.to(dtype)
    softmax_output = torch.nn.functional.softmax(input_tensor, dim=dim)

    if not cpu:
        softmax_output = softmax_output.cpu()

    return {"softmax_output": softmax_output.numpy()}

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
        dim = input.get("dim", None)
        dtype = getattr(tf, input.get("dtype", None)) if input.get("dtype", None) else None

        if dtype:
            input_tensor = tf.cast(input_tensor, dtype)
        # Apply to TensorFlow equivalent
        softmax_output = tf.nn.softmax(input_tensor, axis=dim)

        return {"softmax_output": softmax_output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float32),
        "dim": 1,
        "dtype": None,
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    if np.allclose(torch_result["softmax_output"], tf_result["softmax_output"], atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()