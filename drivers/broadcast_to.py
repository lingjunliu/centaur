import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Ensure computations are done on the correct device
    device = 'cpu' if cpu else 'cuda'
    
    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"]).to(device)
    shape = input["shape"]

    # Apply to torch.broadcast_to
    broadcasted_tensor = torch.broadcast_to(input_tensor, shape)

    if not cpu:
        broadcasted_tensor = broadcasted_tensor.cpu()

    return {"broadcasted_tensor": broadcasted_tensor.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        shape = input["shape"]
        
        # Apply to TensorFlow equivalent
        broadcasted_tensor = tf.broadcast_to(input_tensor, shape)

        return {"broadcasted_tensor": broadcasted_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "shape": (3, 3)
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    assert np.array_equal(torch_result["broadcasted_tensor"], tf_result["broadcasted_tensor"]), "not equal"
    print("equal")

if __name__ == "__main__":
    main()