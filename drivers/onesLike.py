import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    # Create ones_like tensor
    ones_tensor = torch.ones_like(input_tensor, dtype=torch.float32, requires_grad=False)

    if not cpu:
        ones_tensor = ones_tensor.cpu()

    return {"ones_like_tensor": ones_tensor.numpy()}

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
        ones_tensor = tf.ones_like(input_tensor)

        return {"ones_like_tensor": ones_tensor.numpy()}

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

if __name__ == "__main__":
    main()
