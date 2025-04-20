import numpy as np

def torch_version(input, cpu=True):
    import torch
    
    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    dim = input.get("dim", None)

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    # Apply torch.squeeze
    if dim is not None:
        squeezed_tensor = torch.squeeze(input_tensor, dim)
    else:
        squeezed_tensor = torch.squeeze(input_tensor)
    
    if not cpu:
        squeezed_tensor = squeezed_tensor.cpu()

    return {"squeezed_tensor": squeezed_tensor.numpy()}

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

        # Apply TensorFlow equivalent
        if dim is not None:
            squeezed_tensor = tf.squeeze(input_tensor, axis=dim)
        else:
            squeezed_tensor = tf.squeeze(input_tensor)
    
    return {"squeezed_tensor": squeezed_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[[[2], [3]], [[4], [5]]]], dtype=np.float32),
        "dim": None  # You can test with other values like 1, (1, 2), etc.
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assertion to check equality
    np.testing.assert_almost_equal(torch_result["squeezed_tensor"], tf_result["squeezed_tensor"], decimal=5)
    print("equal") if np.allclose(torch_result["squeezed_tensor"], tf_result["squeezed_tensor"]) else print("not equal")

if __name__ == "__main__":
    main()