import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    inplace = input.get("inplace", False)
    
    # Apply torch.nn.Hardsigmoid
    hardsigmoid_layer = torch.nn.Hardsigmoid(inplace=inplace)
    output_tensor = hardsigmoid_layer(input_tensor)

    if not cpu:
        output_tensor = output_tensor.cpu()
    
    return {"hardsigmoid_output": output_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        
        # Apply TensorFlow equivalent for Hardsigmoid
        output_tensor = tf.clip_by_value(input_tensor / 6.0 + 0.5, 0.0, 1.0)
        
        return {"hardsigmoid_output": output_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.randn(2, 3).astype(np.float32),  # Random input
        "inplace": False
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert and compare results
    torch_output = torch_result["hardsigmoid_output"]
    tf_output = tf_result["hardsigmoid_output"]
    
    if np.allclose(torch_output, tf_output, atol=1e-6):
        print("Results are equal")
    else:
        print("Results are not equal")

if __name__ == "__main__":
    main()