import numpy as np

def torch_where_version(input, cpu=True):
    import torch

    # Extract parameters from input
    condition_tensor = torch.tensor(input["condition"], dtype=torch.bool)
    input_tensor = torch.tensor(input.get("input", None)) if input.get("input", None) is not None else None
    other_tensor = torch.tensor(input.get("other", None)) if input.get("other", None) is not None else None

    if not cpu:
        condition_tensor = condition_tensor.cuda()
        if input_tensor is not None:
            input_tensor = input_tensor.cuda()
        if other_tensor is not None:
            other_tensor = other_tensor.cuda()

    if input_tensor is not None and other_tensor is not None:
        result = torch.where(condition_tensor, input_tensor, other_tensor)
    else:
        result = torch.nonzero(condition_tensor, as_tuple=True)

    if not cpu:
        if isinstance(result, torch.Tensor):
            result = result.cpu()
        else:
            for elem in result:
                elem.cpu()

    return {"result": result.numpy() if isinstance(result, torch.Tensor) else [elem.numpy() for elem in result]}

def tensorflow_where_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Extract parameters from input
        condition_tensor = tf.constant(input["condition"], dtype=tf.bool)
        input_tensor = tf.constant(input.get("input", None)) if input.get("input", None) is not None else None
        other_tensor = tf.constant(input.get("other", None)) if input.get("other", None) is not None else None

        if input_tensor is not None and other_tensor is not None:
            result = tf.where(condition_tensor, input_tensor, other_tensor)
        else:
            result = tf.experimental.numpy.nonzero(condition_tensor)

        return {"result": result.numpy() if isinstance(result, tf.Tensor) else [elem.numpy() for elem in result]}

def main():
    # Example input for first case
    input_data1 = {
        "condition": np.array([[True, False, True], [False, True, False]], dtype=bool),
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "other": np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32)
    }

    # Example input for second case
    input_data2 = {
        "condition": np.array([[True, False, True], [False, True, False]], dtype=bool)
    }

    # Torch example for first case
    torch_result1 = torch_where_version(input_data1)
    print("Torch result (case 1):", torch_result1)

    # TensorFlow example for first case
    tf_result1 = tensorflow_where_version(input_data1)
    print("TensorFlow result (case 1):", tf_result1)

    # Torch example for second case
    torch_result2 = torch_where_version(input_data2)
    print("Torch result (case 2):", torch_result2)

    # TensorFlow example for second case
    tf_result2 = tensorflow_where_version(input_data2)
    print("TensorFlow result (case 2):", tf_result2)

    # Compare results for first case
    assert np.array_equal(torch_result1["result"], tf_result1["result"]), "Results are not equal for case 1"
    print("equal")

    # Compare results for second case
    assert all(np.array_equal(tr, tfr) for tr, tfr in zip(torch_result2["result"], tf_result2["result"])), "Results are not equal for case 2"
    print("equal")

if __name__ == "__main__":
    main()