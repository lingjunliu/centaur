import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    reduce_indices = torch.tensor(input_dict["reduce_indices"])
    op = input_dict.get("op", "sum")
    initial = input_dict.get("initial", None)

    if not cpu:
        input_tensor = input_tensor.cuda()
        reduce_indices = reduce_indices.cuda()

    if initial is not None:
        initial = torch.tensor(initial)
        if not cpu:
            initial = initial.cuda()
        result = torch.segment_reduce(input_tensor, reduce_indices, reduce=op, initial=initial)
    else:
        result = torch.segment_reduce(input_tensor, reduce_indices, reduce=op)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input = tf.constant(input_dict["input"])
        reduce_indices = tf.constant(input_dict["reduce_indices"])
        op = input_dict.get("op", "sum")
        initial = input_dict.get("initial", None)

        num_segments = tf.reduce_max(reduce_indices) + 1
        
        if op == "sum":
            result = tf.math.segment_sum(input, reduce_indices)
        elif op == "mean":
            result = tf.math.segment_mean(input, reduce_indices)
        elif op == "max":
            result = tf.math.segment_max(input, reduce_indices)
        elif op == "min":
            result = tf.math.segment_min(input, reduce_indices)
        else:
            raise ValueError(f"Unsupported operation: {op}")

        if initial is not None:
            mask = tf.range(num_segments) < tf.reduce_max(reduce_indices) + 1
            initial_tensor = tf.constant(initial, dtype=result.dtype)
            result = tf.where(mask, result, initial_tensor)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
        "reduce_indices": np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
        "op": "sum",
        "initial": -1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
        "reduce_indices": np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
        "op": "mean"
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
        "reduce_indices": np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
        "op": "max"
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
        "reduce_indices": np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
        "op": "min"
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
        "reduce_indices": np.array([0, 0, 1, 1, 2, 2], dtype=np.int32),
        "op": "sum"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()