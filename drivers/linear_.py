import numpy as np

def torch_version_linear(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    weight_tensor = torch.tensor(input["weight"])
    bias_tensor = torch.tensor(input.get("bias", None)) if input.get("bias", None) is not None else None
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        weight_tensor = weight_tensor.cuda()
        if bias_tensor is not None:
            bias_tensor = bias_tensor.cuda()

    # Apply torch.nn.functional.linear
    output = torch.nn.functional.linear(input_tensor, weight_tensor, bias=bias_tensor)

    if not cpu:
        output = output.cpu()

    return {"linear_output": output.numpy()}

def tensorflow_version_linear(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        weight_tensor = tf.constant(input["weight"])
        bias_tensor = tf.constant(input.get("bias", None)) if input.get("bias", None) is not None else None

        # Apply tf.matmul for TensorFlow equivalent
        output = tf.matmul(input_tensor, tf.transpose(weight_tensor))  # Note the transpose

        if bias_tensor is not None:
            output = tf.add(output, bias_tensor)

        return {"linear_output": output.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "weight": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        "bias": np.array([1.0, 2.0], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version_linear(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version_linear(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    if np.allclose(torch_result["linear_output"], tf_result["linear_output"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()