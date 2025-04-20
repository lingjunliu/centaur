import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Prepare input tensor
    input_tensor = torch.tensor(input["input"])
    
    # Apply torch.deg2rad
    result_tensor = torch.deg2rad(input_tensor)

    if not cpu:
        result_tensor = result_tensor.cpu()
    
    return {"deg2rad_result": result_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Prepare input tensor
        input_tensor = tf.constant(input["input"])

        # Apply TensorFlow equivalent: tf.deg2rad
        result_tensor = tf.experimental.numpy.deg2rad(input_tensor)

        return {"deg2rad_result": result_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[180.0, -180.0], [360.0, -360.0], [90.0, -90.0]], dtype=np.float32),
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare the results
    torch_result_np = torch_result["deg2rad_result"]
    tf_result_np = tf_result["deg2rad_result"]

    if np.allclose(torch_result_np, tf_result_np):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()