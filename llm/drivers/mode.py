import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    dim = input_dict.get("dim", -1)
    keepdim = input_dict.get("keepdim", False)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.mode(input_tensor, dim=dim, keepdim=keepdim)

    if not cpu:
        result = (result.values.cpu(), result.indices.cpu())

    return {"values": result.values.numpy(), "indices": result.indices.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    from collections import Counter

    input_np = input_dict["input"]
    dim = input_dict.get("dim", -1)
    keepdim = input_dict.get("keepdim", False)

    if dim < 0:
        dim = len(input_np.shape) + dim

    input_tensor = tf.convert_to_tensor(input_np)

    axis = dim

    values = []
    indices = []

    if len(input_np.shape) == 1:
        counts = Counter(input_np)
        mode_value = counts.most_common(1)[0][0]
        mode_index = np.where(input_np == mode_value)[0][0]
        values = np.array([mode_value])
        indices = np.array([mode_index])
    else:
        for row in np.ndindex(*input_np.shape[:axis]):
            row_data = input_np[row]
            counts = Counter(row_data.flatten().tolist())
            if counts:
                mode_value = counts.most_common(1)[0][0]
                mode_index = np.where(row_data == mode_value)[0][0]
            else:
                mode_value = 0
                mode_index = 0
            values.append(mode_value)
            indices.append(mode_index)

        values = np.array(values)
        indices = np.array(indices)

        new_shape = input_np.shape[:axis] + input_np.shape[axis+1:]
        if values.size > 0:
            values = values.reshape(new_shape)
            indices = indices.reshape(new_shape)
        else:
            values = np.array([])
            indices = np.array([])

    if keepdim:
        values = np.expand_dims(values, axis=dim)
        indices = np.expand_dims(indices, axis=dim)

    return {"values": values, "indices": indices}


def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[0, 0, 0, 2, 0, 0, 2],
                           [0, 3, 0, 0, 2, 0, 1],
                           [2, 2, 2, 0, 0, 0, 3],
                           [2, 2, 3, 0, 1, 1, 0],
                           [1, 1, 0, 0, 2, 0, 2]], dtype=np.int64),
        "dim": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["values"], tf_result["values"], atol=A_TOL), "Values do not match"
    assert np.allclose(torch_result["indices"], tf_result["indices"], atol=A_TOL), "Indices do not match"

    input_data = {
        "input": np.array([[0, 0, 0, 2, 0, 0, 2],
                           [0, 3, 0, 0, 2, 0, 1],
                           [2, 2, 2, 0, 0, 0, 3],
                           [2, 2, 3, 0, 1, 1, 0],
                           [1, 1, 0, 0, 2, 0, 2]], dtype=np.int64),
        "dim": 0,
        "keepdim": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["values"], tf_result["values"], atol=A_TOL), "Values do not match"
    assert np.allclose(torch_result["indices"], tf_result["indices"], atol=A_TOL), "Indices do not match"


    print("Success")

if __name__ == "__main__":
    main()