import numpy as np

# Ensure that you have a set_seed function
def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    
    # Apply torch.msort
    sorted_tensor = torch.msort(input_tensor)

    if not cpu:
        sorted_tensor = sorted_tensor.cpu()

    return {"sorted_tensor": sorted_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        
        # TensorFlow equivalent to sort along the first dimension
        sorted_tensor = tf.sort(input_tensor, axis=0)

        return {"sorted_tensor": sorted_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[-0.1321, 0.4370, -1.2631, -1.1289], [-2.0527, -1.1250, 0.2275, 0.3077], [-0.0881, -0.1259, -0.5495, 1.0284]], dtype=np.float32),
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert and compare
    assert np.allclose(torch_result["sorted_tensor"], tf_result["sorted_tensor"]), "Results are not equal"
    print("equal")

if __name__ == "__main__":
    main()