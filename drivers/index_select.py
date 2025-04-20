import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    dim = input["dim"]
    index = torch.tensor(input["index"], dtype=torch.int64)

    if not cpu:
        input_tensor = input_tensor.cuda()
        index = index.cuda()

    # Apply to torch.index_select
    result_tensor = torch.index_select(input_tensor, dim, index)

    if not cpu:
        result_tensor = result_tensor.cpu()

    return { "result": result_tensor.numpy() }

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/CPU:0"
    else:
        device_string = "/GPU:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        dim = input["dim"]
        index = tf.constant(input["index"], dtype=tf.int32)

        # Apply to TensorFlow equivalent (tf.gather)
        if dim == 0:
            result_tensor = tf.gather(input_tensor, index, axis=dim)
        else:
            result_tensor = tf.gather(input_tensor, index, axis=dim)

        return { "result": result_tensor.numpy() }

def main():
    # Example input
    input_data = {
        "input": np.array([[0.5, 0.3, 0.8], [0.2, 0.6, 0.9]], dtype=np.float32),
        "dim": 1,
        "index": np.array([2, 0], dtype=np.int64)
    }

    # Torch example
    torch_result = torch_version(input_data)
    #print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    #print("TensorFlow result:", tf_result)

    if np.array_equal(torch_result, tf_result):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()