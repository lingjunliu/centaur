import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    # Apply to torch.nn.functional.logsigmoid
    output = torch.nn.functional.logsigmoid(input_tensor)

    if not cpu:
        output = output.cpu()

    return {"logsigmoid_output": output.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # Apply to TensorFlow equivalent
        output = tf.math.log_sigmoid(input_tensor)

        return {"logsigmoid_output": output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([0.5, -0.3, 0.8, -0.2, 0.6, -0.9], dtype=np.float32),
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    if np.allclose(torch_result["logsigmoid_output"], tf_result["logsigmoid_output"], atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()