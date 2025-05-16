import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    if cpu:
        tensors = [torch.tensor(t) for t in input["tensors"]]
    else:
        tensors = [torch.tensor(t).cuda() for t in input["tensors"]]

    result = torch.block_diag(*tensors)

    if not cpu:
        result = result.cpu()

    return {"block_diag_result": result.numpy()}

def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        tensors = [tf.convert_to_tensor(t) for t in input["tensors"]]

        def block_diag(*args):
            if not args:
                return tf.constant([], shape=[0, 0])

            rows = [tf.shape(arg)[0] for arg in args]
            cols = [tf.shape(arg)[1] for arg in args]
            total_row = tf.add_n(rows)
            total_col = tf.add_n(cols)
            result = tf.zeros((total_row, total_col), dtype=args[0].dtype)
            current_row, current_col = 0, 0
            for i, arg in enumerate(args):
                rr, cc = rows[i], cols[i]
                result = tf.tensor_scatter_nd_update(result,
                                                     tf.stack([tf.range(current_row, current_row + rr)[:, None] + tf.zeros([1, cc], dtype=tf.int32),
                                                               tf.range(current_col, current_col + cc)[None, :] + tf.zeros([rr, 1], dtype=tf.int32)], axis=-1),
                                                     arg)
                current_row += rr
                current_col += cc
            return result

        result = block_diag(*tensors)
        return {"block_diag_result": result.numpy()}

def main():
    input_data = {
        "tensors": [
            np.random.rand(2, 2).astype(np.float32),
            np.random.rand(3, 3).astype(np.float32),
            np.random.rand(1, 1).astype(np.float32)
        ]
    }

    torch_result = torch_version(input_data)
    print("Torch result:", torch_result["block_diag_result"])

    tf_result = tensorflow_version(input_data)
    print("TensorFlow result:", tf_result["block_diag_result"])

    # Convert results to numpy arrays for comparison
    torch_np_result = np.array(torch_result["block_diag_result"])
    tf_np_result = np.array(tf_result["block_diag_result"])

    if np.allclose(torch_np_result, tf_np_result, atol=1e-6):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()