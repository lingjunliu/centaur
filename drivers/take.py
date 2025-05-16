import numpy as np

def torch_take_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    index_tensor = torch.tensor(input["index"], dtype=torch.int64)

    # Apply torch.take
    result = torch.take(input_tensor, index_tensor)

    if not cpu:
        result = result.cpu()

    return {"take_result": result.numpy()}

def tensorflow_take_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        index_tensor = tf.constant(input["index"], dtype=tf.int64)

        # Flatten input tensor (since torch.take flattens the input tensor)
        input_tensor_flat = tf.reshape(input_tensor, [-1])

        # Apply tf.gather (equivalent to torch.take)
        result = tf.gather(input_tensor_flat, index_tensor)

        return {"take_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "input": np.array([[4, 3, 5], [6, 7, 8]], dtype=np.float32),
        "index": np.array([0, 2, 5], dtype=np.int64)
    }

    # Torch example
    torch_result = torch_take_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_take_version(input_data)
    print("TensorFlow result:", tf_result)
    
    # Assert and compare results
    if np.array_equal(torch_result["take_result"], tf_result["take_result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()