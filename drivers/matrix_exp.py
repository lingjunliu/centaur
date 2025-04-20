import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["A"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply torch.matrix_exp
    loss = torch.matrix_exp(input_tensor)
    
    if not cpu:
        loss = loss.cpu()
    
    return {"matrix_exp_result": loss.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["A"])
        
        # Apply TensorFlow equivalent
        loss = tf.linalg.expm(input_tensor)

        return {"matrix_exp_result": loss.numpy()}

def main():
    # Example input
    input_data = {
        "A": np.array([[[0.0, 1.0], [-1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Convert results to numpy arrays for comparison
    torch_matrix_exp_result = torch_result["matrix_exp_result"]
    tf_matrix_exp_result = tf_result["matrix_exp_result"]

    # Compare the results
    if np.allclose(torch_matrix_exp_result, tf_matrix_exp_result):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()