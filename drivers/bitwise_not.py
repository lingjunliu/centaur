import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply torch.bitwise_not
    result = torch.bitwise_not(input_tensor)
    
    if not cpu:  # Ensuring this logic to be consistent with potential GPU usage
        result = result.cpu()

    return {"bitwise_not_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # Apply TensorFlow equivalent (bitwise_invert)
        result = tf.bitwise.invert(input_tensor)

        return {"bitwise_not_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([0, 1, -1, 2, -2], dtype=np.int32),
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_output = np.array(torch_result["bitwise_not_result"])
    tf_output = np.array(tf_result["bitwise_not_result"])

    if np.array_equal(torch_output, tf_output):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()