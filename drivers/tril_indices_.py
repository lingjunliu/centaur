import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    # Unpack input dictionary
    rows = input["row"]
    cols = input["col"]
    offset = input.get("offset", 0)
    dtype = torch.tensor(np.array([], dtype=input.get("dtype", np.int64))).dtype

    if cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    # Generating lower triangular indices
    tril_indices = torch.tril_indices(rows, cols, offset=offset, 
                                      dtype=dtype, device=device)

    if not cpu:
        tril_indices = tril_indices.cpu()

    return {"tril_indices": [tril_indices[0].tolist(), tril_indices[1].tolist()]}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Unpack input dictionary
        rows = input["row"]
        cols = input["col"]
        offset = input.get("offset", 0)
        dtype = tf.as_dtype(input["dtype"])

        # Creating dense matrix to extract tril indices
        dense_matrix = np.ones((rows, cols))
        tril_matrix = tf.linalg.band_part(dense_matrix, -1, offset - 1) if offset > 0 else tf.linalg.band_part(dense_matrix, -1, offset)

        # Get the indices where values are non-zero (lower triangular part)
        tril_indices_matrix = tf.where(tf.not_equal(tril_matrix, 0))
        row_indices_tf, col_indices_tf = tf.transpose(tril_indices_matrix)

        if not cpu:
            row_indices_tf = tf.cast(tf.identity(row_indices_tf), dtype)
            col_indices_tf = tf.cast(tf.identity(col_indices_tf), dtype)

        return {"tril_indices": [row_indices_tf.numpy().tolist(), col_indices_tf.numpy().tolist()]}

def main():
    # Example input
    input_data = {
        "row": 3,
        "col": 3,
        "offset": 0,
        "dtype": np.int32,
        "device": 'cpu'
    }

    # Torch example
    torch_result = torch_version(input_data)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result)

    # Comparison logic
    if np.array_equal(torch_result["tril_indices"], tf_result["tril_indices"]):
        print("equal")
    else:
        print("not equal")
        
if __name__ == "__main__":
    main()