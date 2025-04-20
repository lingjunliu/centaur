import numpy as np


def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.poisson(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"poisson": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        
        # Create ones_like tensor
        result = tf.random.poisson(shape=[], lam=input_tensor, dtype=input_tensor.dtype)

        return {"poisson": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    np.testing.assert_allclose(torch_result["poisson"], tf_result["poisson"], rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    main()
