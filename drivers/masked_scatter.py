import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    mask = torch.tensor(input_dict["mask"])
    source = torch.tensor(input_dict["source"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        mask = mask.cuda()
        source = source.cuda()

    result = torch.masked_scatter(input_tensor, mask, source)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        mask = tf.constant(input_dict["mask"])
        source = tf.constant(input_dict["source"])

        input_shape = tf.shape(input_tensor)
        mask_flat = tf.reshape(mask, [-1])
        source_flat = tf.reshape(source, [-1])
        input_flat = tf.reshape(input_tensor, [-1])

        true_indices = tf.where(mask_flat)
        num_true = tf.shape(true_indices)[0]

        source_indices = tf.range(num_true) % tf.shape(source_flat)[0]
        updates = tf.gather(source_flat, source_indices)

        indices = tf.reshape(tf.gather_nd(tf.cast(tf.range(tf.size(mask_flat)), dtype=tf.int64), true_indices), [-1, 1])

        sparse_delta = tf.scatter_nd(indices, updates, shape=[tf.size(mask_flat)])
        result_flat = tf.where(mask_flat, sparse_delta, tf.cast(input_flat, dtype=updates.dtype))
        result = tf.reshape(result_flat, input_shape)
        
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "mask": np.array([[True, False], [False, True]], dtype=np.bool_),
        "source": np.array([5, 6, 7], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()