import numpy as np

def torch_version(input, cpu=True):
    import torch
    
    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"], dtype=torch.bool)
    other_tensor = torch.tensor(input["other"], dtype=torch.bool)
    
    # Apply to torch.logical_xor
    result = torch.logical_xor(input_tensor, other_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"logical_xor_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    
    device_string = "/cpu:0" if cpu else "/gpu:0"
    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"], dtype=tf.bool)
        other_tensor = tf.constant(input["other"], dtype=tf.bool)
        
        # Apply to tf.math.logical_xor
        result = tf.math.logical_xor(input_tensor, other_tensor)
        
        return {"logical_xor_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[True, False, True], [False, True, False]], dtype=np.bool_),
        "other": np.array([[False, False, True], [True, True, False]], dtype=np.bool_)
    }
    
    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)
    
    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)
    
    # Compare results
    torch_result_array = torch_result["logical_xor_result"]
    tf_result_array = tf_result["logical_xor_result"]
    
    if np.array_equal(torch_result_array, tf_result_array):
        print("equal")
    else:
        print("not equal")
    
if __name__ == "__main__":
    main()