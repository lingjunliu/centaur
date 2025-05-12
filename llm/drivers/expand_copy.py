import numpy as np
import tensorflow as tf

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    size = tuple(input_dict["size"])

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.expand_copy(input_tensor, size)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    input_tensor = tf.constant(input_dict["input"])
    size = input_dict["size"]

    input_shape = tf.shape(input_tensor)
    input_rank = tf.rank(input_tensor)
    target_rank = len(size)

    if target_rank < input_rank:
        raise ValueError("Number of dimensions of target size must be >= input tensor")

    diff = target_rank - input_rank
    new_shape_list = [1] * diff + tf.unstack(input_shape)
    new_shape = []
    for s in new_shape_list:
        if isinstance(s, tf.Tensor):
            s_val = tf.get_static_value(s)
            if s_val is not None:
                new_shape.append(s_val)
            else:
                # Handle the case where static value is not available
                # Fallback: Use dynamic shape if static shape can't be inferred
                new_shape.append(s)
        else:
            new_shape.append(s)

    multiples = []
    reshape_shape = []
    for s, new_s in zip(size, new_shape):
        if isinstance(new_s, int):
            multiples.append(s // new_s)
            reshape_shape.append(new_s)

        else:
            # Dynamic shape
            multiples.append(s // tf.cast(new_s, tf.int32))
            reshape_shape.append(new_s)

    tiled_tensor = tf.tile(tf.reshape(input_tensor, reshape_shape), multiples)

    result = tiled_tensor.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 3], dtype=np.int32),
        "size": [2, 3]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "size": [2, 2, 2]
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()