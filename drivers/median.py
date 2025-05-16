import numpy as np

def torch_median(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    tensor = torch.tensor(input["input"])

    if not cpu:
        tensor = tensor.cuda()

    if "dim" in input:
        dim = input["dim"]
        keepdim = input.get("keepdim", False)
        values, indices = torch.median(tensor, dim=dim, keepdim=keepdim)
        if not cpu:
            values, indices = values.cpu(), indices.cpu()
        return {"values": values.numpy(), "indices": indices.numpy()}
    else:
        median_value = torch.median(tensor)
        if not cpu:
            median_value = median_value.cpu()
        return {"median": float(median_value.item())}

def tensorflow_median(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        tensor = tf.constant(input["input"])

        if "dim" in input:
            axis = input["dim"]
            keepdims = input.get("keepdim", False)

            sorted_tensor = tf.sort(tensor, axis=axis)
            mid_index = tf.math.floordiv(tf.shape(tensor)[axis], 2)

            if not keepdims:
                return {
                    "values": tf.gather(sorted_tensor, mid_index, axis=axis).numpy(),
                    "indices": mid_index.numpy()
                }
            else:
                expanded_indices = tf.expand_dims(mid_index, axis=-1)
                return {
                    "values": tf.gather(sorted_tensor, expanded_indices, axis=axis).numpy(),
                    "indices": expanded_indices.numpy()
                }

        else:
            sorted_tensor = tf.sort(tf.reshape(tensor, [-1]))
            mid_index = tf.math.floordiv(tf.size(sorted_tensor), 2)
            return {"median": float(sorted_tensor[mid_index].numpy())}

def main():
    input_data1 = {
        "input": np.array([1.0, 3.0, 2.0, 4.0, 5.0], dtype=np.float32)  # Flat tensor
    }
    
    input_data2 = {
        "input": np.array([[1.0, 3.0, 2.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "dim": 1,
        "keepdim": False
    }

    # Torch example 1
    torch_result1 = torch_median(input_data1)
    print("Torch result 1:", torch_result1)

    # TensorFlow example 1
    tf_result1 = tensorflow_median(input_data1)
    print("TensorFlow result 1:", tf_result1)

    assert np.isclose(torch_result1["median"], tf_result1["median"]), "Results are not equal"
    if np.isclose(torch_result1["median"], tf_result1["median"]):
        print("equal")
    else:
        print("not equal")

    # Torch example 2
    torch_result2 = torch_median(input_data2)
    print("Torch result 2:", torch_result2)

    # TensorFlow example 2
    tf_result2 = tensorflow_median(input_data2)
    print("TensorFlow result 2:", tf_result2)

    assert np.allclose(torch_result2["values"], tf_result2["values"]) and np.allclose(torch_result2["indices"], tf_result2["indices"]), "Results are not equal"
    if np.allclose(torch_result2["values"], tf_result2["values"]) and np.allclose(torch_result2["indices"], tf_result2["indices"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()