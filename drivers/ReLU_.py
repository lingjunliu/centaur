import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    inplace = input.get("inplace", False)

    # Move tensor to device if not CPU
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply torch.nn.ReLU
    relu = torch.nn.ReLU(inplace=inplace)
    output_tensor = relu(input_tensor)

    if not cpu:
        output_tensor = output_tensor.cpu()

    return {"relu_output": output_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # Apply tf.nn.relu (TensorFlow does not support inplace operation)
        output_tensor = tf.nn.relu(input_tensor)
        
        return {"relu_output": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(4, 4).astype(np.float32),  # Random 4x4 matrix
        "inplace": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    torch_output = torch_result["relu_output"]
    tf_output = tf_result["relu_output"]
    if np.allclose(torch_output, tf_output, rtol=1e-05, atol=1e-08):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()