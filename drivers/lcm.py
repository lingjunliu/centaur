import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"], dtype=torch.int32)
    other_tensor = torch.tensor(input["other"], dtype=torch.int32)
    
    # Apply torch.lcm
    result = torch.lcm(input_tensor, other_tensor)

    if not cpu:
        result = result.cpu()

    return {"lcm_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
        
    with tf.device(device_string):

        # Unpack input dictionary
        input_tensor = tf.constant(input["input"], dtype=tf.int32)
        other_tensor = tf.constant(input["other"], dtype=tf.int32)

        # Calculate GCD using TensorFlow
        gcd_tensor = tf.experimental.numpy.gcd(input_tensor, other_tensor)

        # Calculate LCM using the relation LCM * GCD = |a * b|
        lcm_tensor = tf.abs(input_tensor * other_tensor) // gcd_tensor

        return {"lcm_result": lcm_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([5, 10, 15], dtype=np.int32),
        "other": np.array([3, 4, 5], dtype=np.int32),
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    if np.array_equal(torch_result["lcm_result"], tf_result["lcm_result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()