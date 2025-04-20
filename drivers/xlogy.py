import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["other"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        other_tensor = other_tensor.cuda()
    
    # Apply torch.special.xlogy (alias for torch.xlogy)
    result_tensor = torch.special.xlogy(input_tensor, other_tensor)
    
    if not cpu:
        result_tensor = result_tensor.cpu()
    
    return {"xlogy_result": result_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        other_tensor = tf.constant(input["other"])
        
        # Apply TensorFlow equivalent
        result_tensor = tf.math.xlogy(input_tensor, other_tensor)
        
        return {"xlogy_result": result_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "other": np.array([[1.0, 0.0, 2.0], [1.0, 3.0, 0.5]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Comparing the results
    torch_xlogy_result = torch_result["xlogy_result"]
    tf_xlogy_result = tf_result["xlogy_result"]

    if np.allclose(torch_xlogy_result, tf_xlogy_result, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()