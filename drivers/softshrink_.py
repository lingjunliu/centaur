import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    lambd = input.get("lambd", 0.5)
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply to torch.nn.functional.softshrink
    loss = torch.nn.functional.softshrink(input_tensor, lambd=lambd)

    if not cpu:
        loss = loss.cpu()

    return {"softshrink_output": loss.numpy()}

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

        # Apply to TensorFlow equivalent (custom implementation)
        def softshrink(x, lambd):
            return tf.where(x > lambd, x - lambd, tf.where(x < -lambd, x + lambd, 0))

        loss = softshrink(input_tensor, lambd)

        return {"softshrink_output": loss.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([0.4, -0.6, 0.8, -1.0, 0.0], dtype=np.float32),
        "lambd": 0.5
    }

    # Torch example
    torch_result = torch_version(input_data, cpu=True)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data, cpu=True)
    print("TensorFlow result:", tf_result)

    # Compare the results
    torch_output = torch_result["softshrink_output"]
    tf_output = tf_result["softshrink_output"]

    if np.allclose(torch_output, tf_output, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()