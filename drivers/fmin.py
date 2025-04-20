import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["other"])

    # Compute the element-wise minimum using torch.fmin
    result = torch.fmin(input_tensor, other_tensor)

    if cpu:
        result = result.cpu()

    return {"fmin_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    # Unpack input dictionary
    device_string = "/cpu:0" if cpu else "/gpu:0"
    with tf.device(device_string):
        input_tensor = tf.constant(input["input"])
        other_tensor = tf.constant(input["other"])
        
        # Use tf.where to simulate fmin behavior
        result = tf.where(tf.math.is_nan(input_tensor), other_tensor, 
                          tf.where(tf.math.is_nan(other_tensor), input_tensor, 
                                   tf.math.minimum(input_tensor, other_tensor)))

    return {"fmin_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([2.2, float('nan'), 2.1, float('nan')], dtype=np.float32),
        "other": np.array([-9.3, 0.1, float('nan'), float('nan')], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert and compare results using numpy's array_equal to handle NaN correctly
    if np.array_equal(torch_result["fmin_result"], tf_result["fmin_result"], equal_nan=True):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()