import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    if cpu:
        tensors = [torch.tensor(t) for t in input["tensors"]]
    else:
        tensors = [torch.tensor(t).cuda() for t in input["tensors"]]
    
    dim = input.get("dim", 0)

    # Apply torch.cat
    result = torch.cat(tensors, dim=dim)

    if not cpu:
        result = result.cpu()

    return { 'output': result.numpy() }

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        tensors = [tf.constant(t) for t in input["tensors"]]
        dim = input.get("dim", 0)

        # Apply tf.concat
        result = tf.concat(tensors, axis=dim)

        return { 'output': result.numpy() }

def main():
    # Example input
    input_data = {
        "tensors": [
            np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
            np.array([[1.5, 2.3, 1.8], [1.2, 1.6, 1.9]], dtype=np.float32)
        ],
        "dim": 0  # Concatenate along the first dimension
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)
    
    # Assertion to compare results
    assert np.array_equal(torch_result['output'], tf_result['output']), "Results are not equal"
    if np.array_equal(torch_result['output'], tf_result['output']):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()