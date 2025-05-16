import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    offsets = torch.tensor(input_dict["offsets"])
    indices = torch.tensor(input_dict["indices"])
    sparse = input_dict.get("sparse", False)
    include_last_offset = input_dict.get("include_last_offset", False)
    mode = input_dict.get("mode", "sum")
    if mode == "sum":
        mode_enum = 0
    elif mode == "mean":
        mode_enum = 1
    elif mode == "max":
        mode_enum = 2
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    per_sample_weights = input_dict.get("per_sample_weights", None)
    if per_sample_weights is not None:
        per_sample_weights = torch.tensor(per_sample_weights)
    padding_idx = input_dict.get("padding_idx", None)

    if not cpu:
        input_tensor = input_tensor.cuda()
        offsets = offsets.cuda()
        indices = indices.cuda()
        if per_sample_weights is not None:
            per_sample_weights = per_sample_weights.cuda()

    if per_sample_weights is not None:
        result = torch.embedding_bag(input_tensor, indices, offsets, sparse=sparse, include_last_offset=include_last_offset, mode=mode_enum, per_sample_weights=per_sample_weights)
    else:
        result = torch.embedding_bag(input_tensor, indices, offsets, sparse=sparse, include_last_offset=include_last_offset, mode=mode_enum)

    if not cpu:
        result = result.cpu()

    if isinstance(result, tuple):
        return {"result": result[0].numpy()}
    else:
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
        offsets = tf.constant(input_dict["offsets"])
        indices = tf.constant(input_dict["indices"])
        sparse = input_dict.get("sparse", False)
        include_last_offset = input_dict.get("include_last_offset", False)
        mode = input_dict.get("mode", "sum")
        per_sample_weights = input_dict.get("per_sample_weights", None)
        padding_idx = input_dict.get("padding_idx", None)

        if per_sample_weights is not None:
            per_sample_weights = tf.constant(per_sample_weights)

        num_bags = tf.shape(offsets)[0]
        
        outputs = []
        for i in range(num_bags):
            start = offsets[i]
            if i + 1 < num_bags:
                end = offsets[i+1]
            else:
                end = tf.shape(indices)[0]
            
            bag_indices = indices[start:end]
            bag_embeddings = tf.gather(input_tensor, bag_indices)

            if per_sample_weights is not None:
                bag_weights = per_sample_weights[start:end]
                bag_embeddings = bag_embeddings * tf.expand_dims(bag_weights, axis=-1)
            
            if mode == "sum":
                output = tf.reduce_sum(bag_embeddings, axis=0)
            elif mode == "mean":
                output = tf.reduce_mean(bag_embeddings, axis=0)
            elif mode == "max":
                output = tf.reduce_max(bag_embeddings, axis=0)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            
            outputs.append(output)
            
        result = tf.stack(outputs)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]], dtype=np.float32),
        "indices": np.array([0, 1, 2, 3, 4], dtype=np.int64),
        "offsets": np.array([0, 2, 3], dtype=np.int64),
        "mode": "sum"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()