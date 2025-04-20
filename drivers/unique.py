import numpy as np

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    input_tensor = torch.tensor(input["input"])
    sorted_val = input.get("sorted", True)
    return_inverse = input.get("return_inverse", False)
    return_counts = input.get("return_counts", False)
    dim = input.get("dim", None)

    # Apply to torch.unique
    output = torch.unique(
        input_tensor, sorted=sorted_val, return_inverse=return_inverse,
        return_counts=return_counts, dim=dim
    )

    if not cpu:
        if isinstance(output, tuple):
            output = tuple(t.cpu() for t in output)
        else:
            output = output.cpu()

    result = {}
    if isinstance(output, tuple):
        result["unique"], result["inverse_indices"], result["counts"] = [
            t.numpy() if t is not None else None for t in output]
    else:
        result["unique"] = output.numpy()

    return result

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        input_tensor = tf.constant(input["input"])
        sorted_val = input.get("sorted", True)
        return_inverse = input.get("return_inverse", False)
        return_counts = input.get("return_counts", False)
        dim = input.get("dim", None)

        # Flatten the tensor if `dim` is None
        if dim is None:
            input_tensor = tf.reshape(input_tensor, [-1])

        # Apply to TensorFlow equivalent
        unique_values, idx = tf.unique(input_tensor)

        if return_inverse:
            inverse_indices = idx
        else:
            inverse_indices = None

        if return_counts:
            counts = tf.math.bincount(idx, minlength=tf.shape(unique_values)[0])
        else:
            counts = None

        if sorted_val:
            order = tf.argsort(unique_values)
            unique_values = tf.gather(unique_values, order)
            if return_inverse:
                inverse_indices = tf.gather(order, inverse_indices)
            if return_counts:
                counts = tf.gather(counts, order)

        result = {"unique": unique_values.numpy()}
        if return_inverse:
            result["inverse_indices"] = inverse_indices.numpy()
        if return_counts:
            result["counts"] = counts.numpy()

        return result

def main():
    # Example input
    input_data = {
        "input": np.array([1, 3, 2, 3, 1], dtype=np.float32),
        "sorted": True,
        "return_inverse": True,
        "return_counts": True,
        "dim": None
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Convert outputs to numpy arrays and compare
    torch_unique = np.array(torch_result["unique"])
    tf_unique = np.array(tf_result["unique"])
    torch_inverse_indices = np.array(torch_result["inverse_indices"]) if "inverse_indices" in torch_result else None
    tf_inverse_indices = np.array(tf_result["inverse_indices"]) if "inverse_indices" in tf_result else None
    torch_counts = np.array(torch_result["counts"]) if "counts" in torch_result else None
    tf_counts = np.array(tf_result["counts"]) if "counts" in tf_result else None

    # Assertion and results comparison
    if np.array_equal(torch_unique, tf_unique) and (torch_inverse_indices is None or np.array_equal(torch_inverse_indices, tf_inverse_indices)) and (torch_counts is None or np.array_equal(torch_counts, tf_counts)):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()