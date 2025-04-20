import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    mat2_tensor = torch.tensor(input["mat2"])

    # Move to GPU if needed
    if not cpu:
        input_tensor = input_tensor.cuda()
        mat2_tensor = mat2_tensor.cuda()

    # Apply torch.mm
    result = torch.mm(input_tensor, mat2_tensor)

    # Move result to CPU if needed
    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        mat2_tensor = tf.constant(input["mat2"])

        # Apply tf.matmul
        result = tf.matmul(input_tensor, mat2_tensor)

        return {"result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "mat2": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assertion to compare results
    assert np.allclose(torch_result["result"], tf_result["result"]), "Results are not equal"
    if np.allclose(torch_result["result"], tf_result["result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()