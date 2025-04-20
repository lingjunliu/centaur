import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    lambd = input.get("lambd", 0.5)

    # Apply torch.nn.Hardshrink
    hardshrink = torch.nn.Hardshrink(lambd=lambd)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        hardshrink = hardshrink.cuda()

    result = hardshrink(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"hardshrink_output": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        lambd = input.get("lambd", 0.5)

        # Define the Hardshrink function manually
        def hardshrink(x, lambd):
            return tf.where(tf.abs(x) > lambd, x, tf.zeros_like(x))

        result = hardshrink(input_tensor, lambd)

        return {"hardshrink_output": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([1.0, -0.75, 0.3, -1.5, 0.5, 0.2], dtype=np.float32),
        "lambd": 0.5
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result["hardshrink_output"])

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result["hardshrink_output"])

    # Check if the outputs are the same
    if np.allclose(torch_result["hardshrink_output"], tf_result["hardshrink_output"], atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()