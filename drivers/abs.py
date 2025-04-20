import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply torch.abs function
    output_tensor = torch.abs(input_tensor)

    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"abs_result": output_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # Apply TensorFlow equivalent function
        output_tensor = tf.abs(input_tensor)

        return {"abs_result": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[-1.0, -2.0, 3.0], [4.0, -5.0, -6.0]], dtype=np.float32),
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assertion to check if results are equal
    assert np.array_equal(torch_result["abs_result"], tf_result["abs_result"]), "Results are not equal"
    if np.array_equal(torch_result["abs_result"], tf_result["abs_result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()