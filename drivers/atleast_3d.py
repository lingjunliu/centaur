import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    tensors = input["tensors"]

    # Move tensors to appropriate device if needed
    if not cpu:
        tensors = [torch.tensor(t).cuda() for t in tensors]
    else:
        tensors = [torch.tensor(t) for t in tensors]

    # Apply torch.atleast_3d
    result_list = torch.atleast_3d(*tensors)

    # Bring results to cpu if needed
    if not cpu:
        result_list = [res.cpu() for res in result_list]

    return {"atleast_3d_result": [res.numpy() for res in result_list]}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        tensors = input["tensors"]

        # Convert tensors to tf.constant
        tensors = [tf.constant(t) for t in tensors]

        # Apply TensorFlow equivalent
        result_list = [tf.reshape(tensor, tf.concat([tf.ones([3-len(tensor.shape)], dtype=tf.int32), tf.shape(tensor)[:]], axis=0)) if len(tensor.shape) < 3 else tensor for tensor in tensors]

        return {"atleast_3d_result": [res.numpy() for res in result_list]}

def main():
    # Example input
    input_data = {
        "tensors": [
            np.array([1, 2, 3], dtype=np.float32),
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.float32)
        ]
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assert equality
    torch_result_flat = np.concatenate(torch_result["atleast_3d_result"], axis=None)
    tf_result_flat = np.concatenate(tf_result["atleast_3d_result"], axis=None)
    if np.allclose(torch_result_flat, tf_result_flat):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()