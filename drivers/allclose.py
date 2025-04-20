import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["other"])
    rtol = input.get("rtol", 1e-05)
    atol = input.get("atol", 1e-08)
    equal_nan = input.get("equal_nan", False)
    
    # Check the device
    if not cpu:
        input_tensor = input_tensor.cuda()
        other_tensor = other_tensor.cuda()
    
    # Apply torch.allclose function
    result = torch.allclose(input_tensor, other_tensor, rtol=rtol, atol=atol, equal_nan=equal_nan)
    
    return {"allclose": result}

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
        rtol = input.get("rtol", 1e-05)
        atol = input.get("atol", 1e-08)
        equal_nan = input.get("equal_nan", False)
        
        # Calculate absolute tolerances
        abs_diff = tf.abs(input_tensor - other_tensor)
        tol = atol + rtol * tf.abs(other_tensor)
        
        # Apply TensorFlow equivalent
        result = tf.reduce_all(abs_diff <= tol)
        
        # Handle NaN comparisons if requested
        if equal_nan:
            result = tf.logical_or(result, tf.reduce_all(tf.math.equal(input_tensor, other_tensor)))
        
        return {"allclose": bool(result.numpy())}

def main():
    # Example input
    input_data = {
        "input": np.array([10000., 1e-07], dtype=np.float32),
        "other": np.array([10000.1, 1e-08], dtype=np.float32),
        "rtol": 1e-05,
        "atol": 1e-08,
        "equal_nan": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert to check if results are equal
    if torch.tensor(torch_result["allclose"]) == tf.constant(tf_result["allclose"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()