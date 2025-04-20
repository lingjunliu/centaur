import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    lambd = input.get("lambd", 0.5)
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply to torch.nn.functional.hardshrink
    result = torch.nn.functional.hardshrink(input_tensor, lambd=lambd)

    if not cpu:
        result = result.cpu()

    return {"hardshrink_result": result.numpy()}

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

        # Implement hard shrink function for TensorFlow
        def hardshrink(x, lambd):
            return tf.where(tf.abs(x) > lambd, x, tf.zeros_like(x))

        result = hardshrink(input_tensor, lambd)

        return {"hardshrink_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[-0.7, -0.3, 0.2, 0.8], [0.4, -1.0, 0, 1.5]], dtype=np.float32),
        "lambd": 0.5
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assertion to compare results
    assert np.array_equal(torch_result["hardshrink_result"], tf_result["hardshrink_result"]), "Results do not match"
    print("Results are equal")

if __name__ == "__main__":
    main()