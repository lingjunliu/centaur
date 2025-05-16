import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Convert input to Torch tensor
    input_tensor = torch.tensor(input["input"])
    
    # Apply torch.isinf
    result = torch.isinf(input_tensor)
    
    if not cpu:
        result = result.cpu()

    return {"isinf_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Convert input to TensorFlow tensor
        input_tensor = tf.constant(input["input"])
        
        # Apply TensorFlow equivalent of isinf
        result = tf.math.is_inf(input_tensor)

        return {"isinf_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([1, float('inf'), 2, float('-inf'), float('nan')], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    assert np.array_equal(torch_result["isinf_result"], tf_result["isinf_result"]), "Results are not equal"
    print("equal")

if __name__ == "__main__":
    main()
