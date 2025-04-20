import numpy as np

def torch_roll(input_data, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input_data["input"])
    shifts = input_data["shifts"]
    dims = input_data.get("dims", None)

    # Perform torch roll
    rolled_tensor = torch.roll(input=input_tensor, shifts=shifts, dims=dims)

    if not cpu:
        rolled_tensor = rolled_tensor.cpu()

    return {"rolled_tensor": rolled_tensor.numpy()}

def tensorflow_roll(input_data, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input_data["input"])
        shifts = input_data["shifts"]
        dims = input_data.get("dims", None)
        
        if dims is None:
            input_tensor_flat = tf.reshape(input_tensor, [-1])
            rolled_tensor = tf.roll(input_tensor_flat, shift=shifts, axis=0)
            rolled_tensor = tf.reshape(rolled_tensor, tf.shape(input_tensor))
        else:
            rolled_tensor = tf.roll(input_tensor, shift=shifts, axis=dims)
        
        return {"rolled_tensor": rolled_tensor.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32),
        "shifts": 1,
        "dims": 0
    }

    # Torch example
    torch_result = torch_roll(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_roll(input_data)
    print("TensorFlow result:", tf_result)

    # Convert both outputs to numpy arrays for comparison
    torch_rolled_np = np.array(torch_result["rolled_tensor"])
    tf_rolled_np = np.array(tf_result["rolled_tensor"])

    assert np.allclose(torch_rolled_np, tf_rolled_np), "Results do not match!"
    print("equal" if np.allclose(torch_rolled_np, tf_rolled_np) else "not equal")

if __name__ == "__main__":
    main()