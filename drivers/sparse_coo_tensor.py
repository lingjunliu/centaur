from src.type_mapping_torch import np_to_torch
import numpy as np

def torch_sparse_coo_tensor_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    indices = torch.tensor(input["indices"], dtype=torch.long)
    values = torch.tensor(input["values"])
    if "dtype" in input.keys():
        dtype = np_to_torch(input["dtype"])

    size = input.get("size", None)
    
    device = torch.device("cpu" if cpu else "cuda")
    requires_grad = input.get("requires_grad", False)
    check_invariants = input.get("check_invariants", False)
    is_coalesced = input.get("is_coalesced", None)

    if not cpu:
        indices = indices.cuda()
        values = values.cuda()

    # Create the sparse coo tensor
    if dtype is None or device is None:
        if is_coalesced is None:
            sparse_tensor = torch.sparse_coo_tensor(
                indices, values, size=size, requires_grad=requires_grad, check_invariants=check_invariants
            )
        else:
            sparse_tensor = torch.sparse_coo_tensor(
                indices, values, size=size, requires_grad=requires_grad, check_invariants=check_invariants, is_coalesced=is_coalesced
            )
    else:
        if is_coalesced is None:
            sparse_tensor = torch.sparse_coo_tensor(
                indices, values, size=size, dtype=dtype, device=device, requires_grad=requires_grad, check_invariants=check_invariants
            )
        else:
                sparse_tensor = torch.sparse_coo_tensor(
                indices, values, size=size, dtype=dtype, device=device, requires_grad=requires_grad, check_invariants=check_invariants, is_coalesced=is_coalesced
            )

    if not cpu:
        sparse_tensor = sparse_tensor.cpu()

    return {"sparse_tensor": sparse_tensor.to_dense().numpy()}

def tensorflow_sparse_coo_tensor_version(input, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        indices = tf.constant(input["indices"], dtype=tf.int64)
        values = tf.constant(input["values"])
        size = tf.constant(input["size"], dtype=tf.int64) if input.get("size", None) else None

        # Create the sparse tensor equivalent in TensorFlow
        sparse_tensor = tf.sparse.SparseTensor(indices=tf.transpose(indices), values=values, dense_shape=size)

        return {"sparse_tensor": tf.sparse.to_dense(sparse_tensor).numpy()}

def main():
    # Example input
    input_data = {
        "indices": [[0, 1, 2], [2, 0, 1]],
        "values": [3, 4, 5],
        "size": [3, 3],
        "dtype": np.float32,
        "device": None,
        "requires_grad": False
    }

    # Torch example
    torch_result = torch_sparse_coo_tensor_version(input_data)
    torch_sparse_tensor = torch_result["sparse_tensor"]
    print("Torch sparse tensor:", torch_sparse_tensor)

    # TensorFlow example
    tf_result = tensorflow_sparse_coo_tensor_version(input_data)
    tf_sparse_tensor = tf_result["sparse_tensor"]
    print("TensorFlow sparse tensor:", tf_sparse_tensor)

    if np.allclose(torch_sparse_tensor, tf_sparse_tensor):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()