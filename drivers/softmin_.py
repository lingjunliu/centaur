import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    dim = input.get("dim", None)
    dtype = input.get("dtype", None)

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply to torch.nn.functional.softmin
    if dtype is not None:
        input_tensor = input_tensor.to(dtype)

    result = torch.nn.functional.softmin(input_tensor, dim=dim)

    if not cpu:
        result = result.cpu()

    return {"softmin": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        dim = input.get("dim", None)
        dtype = input.get("dtype", None)

        if dtype is not None:
            input_tensor = tf.cast(input_tensor, dtype)

        # Apply to TensorFlow equivalent
        neg_input_tensor = -input_tensor
        softmax_result = tf.nn.softmax(neg_input_tensor, axis=dim)
        result = softmax_result # softmin is achieved by applying softmax to the negated input

        return {"softmin": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "dim": 1,
        "dtype": None
    }


    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Use numpy arrays for comparison
    np.testing.assert_allclose(
        torch_result["softmin"], 
        tf_result["softmin"], 
        rtol=1e-5, 
        atol=1e-8
    )
    print("equal")

if __name__ == "__main__":
    main()