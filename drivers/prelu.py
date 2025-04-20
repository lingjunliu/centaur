import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    weight_tensor = torch.tensor(input["weight"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        weight_tensor = weight_tensor.cuda()

    # Apply to torch.nn.functional.prelu
    output = torch.nn.functional.prelu(input_tensor, weight_tensor)

    if not cpu:
        output = output.cpu()

    return {"prelu_output": output.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        weight_tensor = tf.constant(input["weight"])

        # Apply equivalent functionality using TensorFlow operations
        pos = tf.nn.relu(input_tensor)
        neg = -weight_tensor * tf.nn.relu(-input_tensor)
        output = pos + neg

        return {"prelu_output": output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, -0.3, 0.8], [-0.2, 0.6, -0.9]], dtype=np.float32),
        "weight": np.array([0.25], dtype=np.float32)  # Example scalar weight
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results using numpy
    torch_output = torch_result["prelu_output"]
    tf_output = tf_result["prelu_output"]

    if np.allclose(torch_output, tf_output, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()