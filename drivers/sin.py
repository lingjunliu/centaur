from src.setseed import set_seed
import torch
import tensorflow as tf
import numpy as np

# Example seed setting function
def set_seed(seed=0):
    torch.manual_seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def torch_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    # Convert inputs to tensor
    input_tensor = torch.tensor(input["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    # Compute the sine
    result = torch.sin(input_tensor)

    if cpu:
        result = result.cpu()

    return {"sin_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    # Set seed for reproducibility
    set_seed()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Convert inputs to tensor
        input_tensor = tf.constant(input["input"])
        
        # Compute the sine
        result = tf.sin(input_tensor)

    return {"sin_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[-0.5461, 0.1347, -2.7266, -0.2746]], dtype=np.float32),
    }

    # Torch example
    torch_result = torch_version(input_data, cpu=True)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data, cpu=True)
    print("TensorFlow result:", tf_result)

    # Compare the results
    if np.allclose(torch_result["sin_result"], tf_result["sin_result"], rtol=1e-5, atol=1e-8):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()