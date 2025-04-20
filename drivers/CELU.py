import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    alpha = input.get("alpha", 1.0)
    inplace = input.get("inplace", False)
    input_tensor = torch.tensor(input["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    activation = torch.nn.CELU(alpha=alpha, inplace=inplace)
    result = activation(input_tensor)
    
    if not cpu:
        result = result.cpu()

    return {"celu_activation_output": result.detach().numpy()}


def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        alpha = input.get("alpha", 1.0)
        input_tensor = tf.constant(input["input"])

        condition = tf.greater_equal(input_tensor, 0)
        output_pos = tf.where(condition, input_tensor, tf.zeros_like(input_tensor))
        output_neg = tf.where(condition, tf.zeros_like(input_tensor), alpha * (tf.exp(input_tensor / alpha) - 1))

        result = output_pos + output_neg

        return {"celu_activation_output": result.numpy()}


def main():
    # Example input
    input_data = {
        "input": np.random.randn(2, 3).astype(np.float32),
        "alpha": 1.0,
        "inplace": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Using numpy.allclose for comparing floating values
    if np.allclose(torch_result["celu_activation_output"], tf_result["celu_activation_output"], atol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()