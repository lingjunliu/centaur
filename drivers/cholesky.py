import numpy as np

# Seed setting function for reproducibility
def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    upper = input.get("upper", False)

    if not cpu:
        input_tensor = input_tensor.to('cuda')
    
    # Apply to torch.cholesky, which is deprecated and will be replaced with torch.linalg.cholesky
    cholesky_result = torch.cholesky(input_tensor, upper=upper)
    
    if not cpu:
        cholesky_result = cholesky_result.cpu()

    return {"cholesky_result": cholesky_result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # Apply the TensorFlow equivalent
        cholesky_result = tf.linalg.cholesky(input_tensor)

        return {"cholesky_result": cholesky_result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.random.rand(3, 3).astype(np.float32),
        "upper": False
    }
    input_data["input"] = np.dot(input_data["input"], input_data["input"].T) + 1e-3 * np.eye(3)  # Make symmetric positive-definite matrix

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Comparing both outputs
    torch_output = torch_result["cholesky_result"]
    tf_output = tf_result["cholesky_result"]
    
    if np.allclose(torch_output, tf_output, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()