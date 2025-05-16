import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    embeddings = torch.tensor(input_dict["embeddings"])
    offsets = torch.tensor(input_dict["offsets"])
    indices = torch.tensor(input_dict["indices"])
    mode = input_dict.get("mode", "sum")
    per_sample_weights = input_dict.get("per_sample_weights", None)
    include_last_offset = input_dict.get("include_last_offset", False)

    if not cpu:
        embeddings = embeddings.cuda()
        offsets = offsets.cuda()
        indices = indices.cuda()
        if per_sample_weights is not None:
            per_sample_weights = torch.tensor(per_sample_weights).cuda()

    if per_sample_weights is None:
        result = torch.nn.functional.embedding_bag(
            indices,
            embeddings,
            offsets,
            mode=mode,
            include_last_offset=include_last_offset
        )
    else:
        result = torch.nn.functional.embedding_bag(
            indices,
            embeddings,
            offsets,
            mode=mode,
            per_sample_weights=per_sample_weights,
            include_last_offset=include_last_offset
        )

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    embeddings = tf.constant(input_dict["embeddings"])
    offsets = tf.constant(input_dict["offsets"])
    indices = tf.constant(input_dict["indices"])
    mode = input_dict.get("mode", "sum")
    per_sample_weights = input_dict.get("per_sample_weights", None)
    include_last_offset = input_dict.get("include_last_offset", False)

    if mode != "sum":
        raise ValueError("TensorFlow version only supports 'sum' mode")
    if per_sample_weights is not None:
        raise ValueError("TensorFlow version does not support per_sample_weights")
    if include_last_offset:
        raise ValueError("TensorFlow version does not support include_last_offset")

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        num_bags = tf.shape(offsets)[0]

        result_list = []
        for i in range(num_bags):
            start_index = offsets[i]
            if i + 1 < num_bags:
                end_index = offsets[i + 1]
            else:
                end_index = tf.shape(indices)[0]

            bag_indices = indices[start_index:end_index]
            bag_embeddings = tf.gather(embeddings, bag_indices)
            result_list.append(tf.reduce_sum(bag_embeddings, axis=0))

        result = tf.stack(result_list)
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "embeddings": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        "offsets": np.array([0, 2], dtype=np.int64),
        "indices": np.array([0, 1, 2, 3], dtype=np.int64),
        "mode": "sum"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()