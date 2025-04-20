import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary and convert to tensor
    input_tensor = torch.tensor(input["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    # Apply torch.fliplr
    result = torch.fliplr(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"fliplr_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary and convert to tensor
        input_tensor = tf.convert_to_tensor(input["input"])

        # Flip the tensor left/right using TensorFlow equivalent
        result = tf.reverse(input_tensor, axis=[-1])

    return {"fliplr_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32)
    }

    # Torch example
    torch_result = torch_version(input_data, cpu=True)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data, cpu=True)
    print("TensorFlow result:", tf_result)

    # Assert and print comparison result
    assert np.array_equal(torch_result["fliplr_result"], tf_result["fliplr_result"]), "Results do not match!"
    print("equal")

if __name__ == "__main__":
    main()