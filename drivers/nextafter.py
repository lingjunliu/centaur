import numpy as np

def torch_nextafter_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    
    input_tensor = torch.tensor(input["input"])
    other_tensor = torch.tensor(input["other"])

    result = torch.nextafter(input_tensor, other_tensor)
    
    if not cpu:
        result = result.cpu()
        
    return {"nextafter": result.numpy()}

def tensorflow_nextafter_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input["input"])
        other_tensor = tf.constant(input["other"])

        result = tf.experimental.numpy.nextafter(input_tensor, other_tensor)
        
        return {"nextafter": result.numpy()}

def main():
    input_data = {
        "input": np.array([1.0, 2.0], dtype=np.float32),
        "other": np.array([2.0, 1.0], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_nextafter_version(input_data)
    print("Torch result:", torch_result)
    
    # TensorFlow example
    tf_result = tensorflow_nextafter_version(input_data)
    print("TensorFlow result:", tf_result)
    
    # Compare the results
    torch_result_np = np.array(torch_result["nextafter"])
    tf_result_np = np.array(tf_result["nextafter"])
    
    if np.allclose(torch_result_np, tf_result_np):
        print("equal")
    else:
        print("not equal")
        
    assert np.array_equal(torch_result_np, tf_result_np), "Results are not equal"

if __name__ == "__main__":
    main()