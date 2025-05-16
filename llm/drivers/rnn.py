import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from torch.nn.utils.rnn import pack_padded_sequence

    input_tensor = torch.tensor(input_dict["input"])
    lengths = torch.tensor(input_dict["lengths"])
    batch_first = input_dict.get("batch_first", False)
    enforce_sorted = input_dict.get("enforce_sorted", True)

    if not cpu:
        input_tensor = input_tensor.cuda()
        lengths = lengths.cuda()
    
    result = pack_padded_sequence(input_tensor, lengths, batch_first=batch_first, enforce_sorted=enforce_sorted)
    
    if not cpu:
        result = (result.data.cpu(), result.batch_sizes.cpu(), result.sorted_indices.cpu() if hasattr(result, 'sorted_indices') and result.sorted_indices is not None else None, result.unsorted_indices.cpu() if hasattr(result, 'unsorted_indices') and result.unsorted_indices is not None else None)
    else:
        result = (result.data.numpy(), result.batch_sizes.numpy(), result.sorted_indices.numpy() if hasattr(result, 'sorted_indices') and result.sorted_indices is not None else None, result.unsorted_indices.numpy() if hasattr(result, 'unsorted_indices') and result.unsorted_indices is not None else None)
    
    return {"data": result[0], "batch_sizes": result[1], "sorted_indices": result[2], "unsorted_indices": result[3]}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    input_tensor = tf.constant(input_dict["input"])
    lengths = tf.constant(input_dict["lengths"])
    batch_first = input_dict.get("batch_first", False)
    enforce_sorted = input_dict.get("enforce_sorted", True)

    if enforce_sorted:
        sorted_indices = tf.argsort(lengths, direction='DESCENDING')
        unsorted_indices = tf.argsort(sorted_indices)
        sorted_lengths = tf.gather(lengths, sorted_indices)
        sorted_input = tf.gather(input_tensor, sorted_indices)
    else:
        sorted_lengths = lengths
        sorted_input = input_tensor
        sorted_indices = None
        unsorted_indices = None

    max_length = tf.reduce_max(sorted_lengths)
    mask = tf.sequence_mask(sorted_lengths, maxlen=max_length)
    batch_sizes = tf.reduce_sum(tf.cast(mask, dtype=tf.int32), axis=1)

    if batch_first:
        num_rows = tf.shape(sorted_input)[0]
        num_cols = tf.shape(sorted_input)[1]

        row_indices = tf.range(num_rows)
        col_indices = tf.range(num_cols)
        grid = tf.meshgrid(row_indices, col_indices)
        row_indices_expanded = tf.transpose(grid[0])
        col_indices_expanded = tf.transpose(grid[1])

        valid_indices = tf.transpose(tf.stack([row_indices_expanded, col_indices_expanded]))

        valid_mask = mask

        valid_indices_reshaped = tf.reshape(valid_indices, [-1, 2])
        valid_mask_reshaped = tf.reshape(valid_mask, [-1])

        indices_to_select = tf.boolean_mask(valid_indices_reshaped, valid_mask_reshaped)

        masked_data = tf.gather_nd(sorted_input, indices_to_select)

    else:
        mask = tf.transpose(mask, perm=[1, 0])
        sorted_input = tf.transpose(sorted_input, perm=[1, 0, 2] if len(sorted_input.shape) == 3 else [1, 0])
        masked_data = tf.boolean_mask(sorted_input, tf.reshape(mask, [-1]))
        batch_sizes = tf.reduce_sum(tf.cast(mask, dtype=tf.int32), axis=0)

    return {"data": masked_data.numpy(), "batch_sizes": batch_sizes.numpy(), "sorted_indices": sorted_indices.numpy() if sorted_indices is not None else None, "unsorted_indices": unsorted_indices.numpy() if unsorted_indices is not None else None}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([
            [[1, 2], [3, 4], [0, 0]],
            [[5, 6], [7, 8], [9, 10]]
        ], dtype=np.float32),
        "lengths": np.array([3, 2], dtype=np.int32),
        "batch_first": True,
        "enforce_sorted": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["data"], tf_result["data"], atol=A_TOL), "Data mismatch"
    assert np.array_equal(torch_result["batch_sizes"], tf_result["batch_sizes"]), "Batch sizes mismatch"

    if torch_result["sorted_indices"] is not None:
      assert np.allclose(torch_result["sorted_indices"], tf_result["sorted_indices"], atol=A_TOL), "Sorted indices mismatch"
    else:
      assert tf_result["sorted_indices"] is None

    if torch_result["unsorted_indices"] is not None:
      assert np.allclose(torch_result["unsorted_indices"], tf_result["unsorted_indices"], atol=A_TOL), "Unsorted indices mismatch"
    else:
      assert tf_result["unsorted_indices"] is None

    print("Success")

if __name__ == "__main__":
    main()