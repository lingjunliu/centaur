import numpy as np

def torch_bucketize_version(input, cpu=True):
    import torch

    # Unpack input tensor
    input_tensor = torch.tensor(input["input"])
    boundaries_tensor = torch.tensor(input["boundaries"])
    out_int32 = input.get("out_int32", False)
    right = input.get("right", False)

    # Apply torch.bucketize
    result = torch.bucketize(input_tensor, boundaries_tensor, out_int32=out_int32, right=right)

    if not cpu:
        result = result.cpu()

    return {"bucketize_result": result.numpy()}

# TensorFlow version
def tf_bucketize_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input tensor
        input_tensor = tf.constant(input["input"])
        boundaries_tensor = tf.constant(input["boundaries"])
        out_int32 = input.get("out_int32", False)
        right = input.get("right", False)

        # Equivalent of torch.bucketize using tf.searchsorted
        # Flatten the input tensor since tf.searchsorted requires 1D array
        flat_input_tensor = tf.reshape(input_tensor, [-1])
        result = tf.searchsorted(boundaries_tensor, flat_input_tensor, side='right' if right else 'left')

        # Reshape the result to the original shape of the input
        result = tf.reshape(result, tf.shape(input_tensor))

        if out_int32:
            result = tf.cast(result, tf.int32)
        else:
            result = tf.cast(result, tf.int64)

        return {"bucketize_result": result.numpy()}

# Main function
def main():
    # Example input
    input_data = {
        "input": np.array([[3, 6, 9], [3, 6, 9]], dtype=np.float32),
        "boundaries": np.array([1, 3, 5, 7, 9], dtype=np.float32),
        "out_int32": False,
        "right": False
    }

    # Torch example
    torch_result = torch_bucketize_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tf_bucketize_version(input_data)
    print("TensorFlow result:", tf_result)

    # Compare results
    if np.array_equal(torch_result["bucketize_result"], tf_result["bucketize_result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()
