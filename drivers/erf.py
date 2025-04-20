import numpy as np

def torch_erf_version(input, cpu=True):
    import torch
    
    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    
    # Apply to torch.erf
    output_tensor = torch.erf(input_tensor)

    if not cpu:
        output_tensor = output_tensor.cpu()
    
    return {"erf_output": output_tensor.numpy()}

def tensorflow_erf_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        
        # Apply to TensorFlow equivalent using tf.math.erf
        output_tensor = tf.math.erf(input_tensor)

        return {"erf_output": output_tensor.numpy()}
    

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, -0.3, 0.8], [-0.2, 0.6, 0.9]], dtype=np.float32)
    }
    
    # Torch example
    torch_result = torch_erf_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_erf_version(input_data)
    print("TensorFlow result:", tf_result)
    
    # Convert results to comparable format (numpy arrays)
    torch_output = np.array(torch_result["erf_output"])
    tf_output = np.array(tf_result["erf_output"])

    np.testing.assert_allclose(torch_output, tf_output, rtol=1e-5, atol=1e-8)
    print("equal")

if __name__ == "__main__":
    main()