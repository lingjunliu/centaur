import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    # Ensure tensor is on the correct device
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply PyTorch tanhshrink
    result = torch.nn.Tanhshrink()(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"tanhshrink_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # Apply TensorFlow tanhshrink equivalent
        result = input_tensor - tf.math.tanh(input_tensor)

        return {"tanhshrink_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(2, 3).tolist(),  # Generating a 2x3 matrix with random values
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results using numpy's allclose method for numerical stability
    if np.allclose(torch_result["tanhshrink_result"], tf_result["tanhshrink_result"], rtol=1e-6, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()