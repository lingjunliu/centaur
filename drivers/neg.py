import numpy as np

# Define the PyTorch version
def torch_version(input, cpu=True):
    import torch

    input_tensor = torch.tensor(input["input"])
    if not cpu:
        input_tensor = input_tensor.cuda()

    neg_tensor = torch.neg(input_tensor)
    
    if not cpu:
        neg_tensor = neg_tensor.cpu()

    return {"neg": neg_tensor.numpy()}

# Define the TensorFlow version
def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input["input"])
        neg_tensor = tf.negative(input_tensor)

    return {"neg": neg_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([0.5, -0.3, 0.8, -0.2, 1.0], dtype=np.float32),
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Ensure comparison is done on numpy arrays
    if np.allclose(torch_result["neg"], tf_result["neg"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()