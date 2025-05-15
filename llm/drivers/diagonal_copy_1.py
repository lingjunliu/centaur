import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    src_tensor = torch.tensor(input_dict["src"])
    offset = input_dict.get("offset", 0)
    dim1 = input_dict.get("dim1", 0)
    dim2 = input_dict.get("dim2", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()
        src_tensor = src_tensor.cuda()

    torch.diagonal_copy(input_tensor, src_tensor, offset=int(offset), dim1=int(dim1), dim2=int(dim2))

    if not cpu:
        input_tensor = input_tensor.cpu()

    return {"result": input_tensor.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.Variable(input_dict["input"])
        src_tensor = tf.constant(input_dict["src"])
        offset = int(input_dict.get("offset", 0))
        dim1 = int(input_dict.get("dim1", 0))
        dim2 = int(input_dict.get("dim2", 1))

        input_shape = input_tensor.shape
        src_shape = src_tensor.shape

        dim1 = dim1 % len(input_shape)
        dim2 = dim2 % len(input_shape)

        if dim1 == dim2:
            raise ValueError("dim1 and dim2 cannot be the same")

        diag_len = min(input_shape[dim1], input_shape[dim2] + offset if offset > 0 else input_shape[dim2] - abs(offset))
        
        if len(src_shape) != 1 or src_shape[0] < diag_len:
            raise ValueError("src must have length equal to the diagonal")

        indices = []
        updates = []
        
        for i in range(diag_len):
            index = [0] * len(input_shape)
            index[dim1] = i
            index[dim2] = i + offset
            
            if index[dim2] < 0 or index[dim2] >= input_shape[dim2]:
                continue
            
            indices.append(index)
            updates.append(src_tensor[i])

        indices = tf.constant(indices, dtype=tf.int32)
        updates = tf.constant(updates)

        result = tf.tensor_scatter_nd_update(input_tensor, indices, updates).numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "input": np.zeros((3, 4), dtype=np.float32),
        "src": np.array([1, 2, 3], dtype=np.float32),
        "offset": 0,
        "dim1": 0,
        "dim2": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()