import numpy as np

def torch_logical_not(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Convert the input to a PyTorch Tensor
    input_tensor = torch.tensor(input["input"], dtype=torch.bool)
    
    # Perform logical NOT operation using PyTorch
    result_tensor = torch.logical_not(input_tensor)
    
    if cpu:
        result_tensor = result_tensor.cpu()

    return {"logical_not_result": result_tensor.numpy()}

def tensorflow_logical_not(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Convert the input to a TensorFlow Tensor
        input_tensor = tf.constant(input["input"], dtype=tf.bool)
        
        # Perform logical NOT operation using TensorFlow
        result_tensor = tf.logical_not(input_tensor)
        
        return {"logical_not_result": result_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
    }

    # PyTorch example
    torch_result = torch_logical_not(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_logical_not(input_data)
    print("TensorFlow result:", tf_result)
    
    # Conversion to numpy arrays for comparison
    torch_result_array = np.array(torch_result["logical_not_result"])
    tf_result_array = np.array(tf_result["logical_not_result"])

    if np.array_equal(torch_result_array, tf_result_array):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()