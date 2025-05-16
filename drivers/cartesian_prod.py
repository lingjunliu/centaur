import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    if cpu:
        tensors = [torch.tensor(arr) for arr in input["tensors"]]
    else:
        tensors = [torch.tensor(arr).cuda() for arr in input["tensors"]]

    # Apply `torch.cartesian_prod`
    result = torch.cartesian_prod(*tensors)

    # Move to CPU if needed
    if not cpu:
        result = result.cpu()

    return {"cartesian_prod_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        tensors = [tf.convert_to_tensor(arr) for arr in input["tensors"]]

        # Compute cartesian product in TensorFlow
        result = tf.stack(tf.meshgrid(*tensors, indexing='ij'), axis=-1)
        shape = [-1, len(tensors)]
        result = tf.reshape(result, shape)

        return {"cartesian_prod_result": result.numpy()}

def main():
    # Example input
    input_data = {
        "tensors": [
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0, 4.0], dtype=np.float32)
        ]
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Assertion to check if results are equal
    if np.array_equal(torch_result["cartesian_prod_result"], tf_result["cartesian_prod_result"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()