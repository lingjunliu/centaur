import numpy as np
import tensorflow as tf
import torch

def torch_version(input_dict, cpu=True):
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
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.convert_to_tensor(input_dict["input"])
        dim = input_dict.get("dim", -1)
        keepdim = input_dict.get("keepdim", False)

        if dim == -1:
            dim = len(input_tensor.shape) - 1

        values = []
        indices = []

        if len(input_tensor.shape) == 0:
            values = input_tensor.numpy()
            indices = 0

        elif len(input_tensor.shape) == 1:
            counts = {}
            unique_values = np.unique(input_tensor.numpy())
            for val in unique_values:
                counts[val] = np.sum(input_tensor.numpy() == val)
            mode_value = max(counts, key=counts.get)
            mode_index = np.where(input_tensor.numpy() == mode_value)[0][0]

            values = np.array(mode_value)
            indices = np.array(mode_index)
        else:
            if dim == 0:
                for i in range(input_tensor.shape[1]):
                    column = input_tensor[:, i]
                    counts = {}
                    unique_values = np.unique(column.numpy())
                    for val in unique_values:
                        counts[val] = np.sum(column.numpy() == val)
                    mode_value = max(counts, key=counts.get)
                    mode_index = np.where(column.numpy() == mode_value)[0][0]
                    values.append(mode_value)
                    indices.append(mode_index)
                values = np.array(values)
                indices = np.array(indices)

            elif dim == 1:
                for i in range(input_tensor.shape[0]):
                    row_tensor = input_tensor[i, :]
                    counts = {}
                    unique_values = np.unique(row_tensor.numpy())
                    for val in unique_values:
                        counts[val] = np.sum(row_tensor.numpy() == val)
                    mode_value = max(counts, key=counts.get)
                    mode_index = np.where(row_tensor.numpy() == mode_value)[0][0]
                    values.append(mode_value)
                    indices.append(mode_index)

                values = np.array(values)
                indices = np.array(indices)

            else:
                raise ValueError("Dimension not supported")

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
        "dim": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["values"], tf_result["values"], atol=A_TOL), "Values do not match"
    assert np.allclose(torch_result["indices"], tf_result["indices"], atol=A_TOL), "Indices do not match"

    print("Success")

if __name__ == "__main__":
    main()