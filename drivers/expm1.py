import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])

    # Apply to torch.expm1
    if not cpu and torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
        result = torch.expm1(input_tensor)
        result = result.cpu()
    else:
        result = torch.expm1(input_tensor)

    return {"expm1_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])

        # Apply to TensorFlow equivalent
        result = tf.math.expm1(input_tensor)

        return {"expm1_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare and assert
    torch_expm1_result = torch_result["expm1_result"]
    tf_expm1_result = tf_result["expm1_result"]

    if np.allclose(torch_expm1_result, tf_expm1_result, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()