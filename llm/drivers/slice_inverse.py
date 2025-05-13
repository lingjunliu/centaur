import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    slices = torch.tensor(input_dict["slices"])
    length = input_dict.get("length", None)

    if not cpu:
        input_tensor = input_tensor.cuda()
        slices = slices.cuda()

    result = torch.ops.aten.slice_inverse(input_tensor, slices, 0, None, None, length)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.convert_to_tensor(input_dict["input"])
    slices = input_dict["slices"]
    length = input_dict.get("length", None)

    slices_tf = tf.convert_to_tensor(slices, dtype=tf.int64)

    if length is None:
        length = tf.reduce_sum(tf.cast(slices_tf, dtype=tf.int32))
    else:
        length = tf.convert_to_tensor(length, dtype=tf.int32)

    indices = tf.range(tf.shape(slices_tf)[0])
    segment_ids = tf.repeat(indices, slices_tf)
    result = tf.math.segment_sum(input_tensor, segment_ids)

    padding = tf.maximum(0, length - tf.shape(result)[0])
    result = tf.pad(result, [[0, padding]])

    result = result[:length]
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32),
        "slices": [1, 2, 1, 3],
        "length": 7
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()