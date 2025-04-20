import numpy as np
from src.type_mapping_torch import np_to_torch

def torch_version(input, cpu=True):
    import torch

    # Unpack input dictionary
    row = input["row"]
    col = input["col"]
    offset = input.get("offset", 0)
    dtype = np_to_torch(input.get("dtype", np.int64))

    if cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    # Apply torch.triu_indices
    indices = torch.triu_indices(row, col, offset=offset, dtype=dtype, device=device)

    if not cpu:
        indices = indices.cpu()

    return {"triu_indices": indices.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf

    row = input["row"]
    col = input["col"]
    offset = input.get("offset", 0)
    tf_dtype = tf.as_dtype(input["dtype"])

    # Use CPU or GPU device
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Create rows and columns
        def triu_indices_tensorflow(rows, cols, offset=0, dtype=tf.int64):
            mask = tf.linalg.band_part(tf.ones((rows, cols)), offset, -1)
            indices = tf.where(mask)
            return tf.cast(indices, dtype)

        indices = triu_indices_tensorflow(row, col, offset=offset, dtype=tf_dtype)
        indices = tf.transpose(indices)

        return {"triu_indices": indices.numpy()}

def main():
    # Example input
    input_data = {
        "row": 3,
        "col": 3,
        "offset": 0,
        "dtype": np.int64,
        "device": 'cpu'
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result["triu_indices"])

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result["triu_indices"])

    # Use numpy to compare results
    if np.array_equal(torch_result["triu_indices"], tf_result["triu_indices"]):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()