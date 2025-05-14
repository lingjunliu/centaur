import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    # Apply torch.log2
    result = torch.log2(input_tensor)
    
    if not cpu:
        result = result.cpu()

    return {"log2_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        
        # Apply TensorFlow equivalent of log2
        result = tf.math.log(input_tensor) / tf.math.log(2.0)

        if cpu:
            result = result.numpy()

        return {"log2_result": result}

def main():
    # Example input
    input_data = {
        "input": np.array([0.8419, 0.8003, 0.9971, 0.5287, 0.0490], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("PyTorch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    torch_result_np = np.array(torch_result["log2_result"])
    tf_result_np = np.array(tf_result["log2_result"])

    assert np.allclose(torch_result_np, tf_result_np, atol=1e-5), "Results are not equal"
    if np.allclose(torch_result_np, tf_result_np, atol=1e-5):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()