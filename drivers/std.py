import numpy as np

def torch_version_std(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    dim = input.get("dim", None)
    correction = input.get("correction", 1)
    keepdim = input.get("keepdim", False)
    
    if dim is None:
        std = torch.std(input_tensor, unbiased=(correction == 1))
    else:
        std = torch.std(input_tensor, dim=dim, unbiased=(correction == 1), keepdim=keepdim)

    if not cpu:
        std = std.cpu()

    return std.numpy()

def tensorflow_version_std(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        axis = input.get("dim", None)
        correction = input.get("correction", 1)
        keepdims = input.get("keepdim", False)

        mean = tf.reduce_mean(input_tensor, axis=axis, keepdims=True)
        variance = tf.reduce_mean(tf.square(input_tensor - mean), axis=axis, keepdims=keepdims)
        std = tf.sqrt(variance)

        if correction == 1:
            N = tf.cast(tf.shape(input_tensor)[axis], tf.float32)
            N = tf.maximum(1.0, N)  # Ensure N is at least 1 to avoid division by zero
            correction_factor = tf.sqrt(tf.math.maximum(0.0, N / (N - 1.0)))
            std *= correction_factor

        return std.numpy()

def main():

    input_data_1 = {
        "input": np.random.rand(4, 4).astype(np.float32),
        "dim": 1,
        "correction": 1,
        "keepdim": False
    }
    
    torch_result_1 = torch_version_std(input_data_1)
    tf_result_1 = tensorflow_version_std(input_data_1)
    print("Torch result 1:", torch_result_1)
    print("TensorFlow result 1:", tf_result_1)
    print("equal" if np.allclose(torch_result_1, tf_result_1, rtol=1e-5) else "not equal")

    input_data_2 = {
        "input": np.random.rand(4, 4).astype(np.float32),
        "dim": None,
        "correction": 0,
        "keepdim": True
    }

    torch_result_2 = torch_version_std(input_data_2)
    tf_result_2 = tensorflow_version_std(input_data_2)
    print("Torch result 2:", torch_result_2)
    print("TensorFlow result 2:", tf_result_2)
    print("equal" if np.allclose(torch_result_2, tf_result_2, rtol=1e-5) else "not equal")

if __name__ == "__main__":
    main()
