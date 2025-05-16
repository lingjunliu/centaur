import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    if "offsets" in input_dict:
        offsets = torch.tensor(input_dict["offsets"])
    else:
        offsets = None
    if "per_sample_weights" in input_dict:
        per_sample_weights = torch.tensor(input_dict["per_sample_weights"])
    else:
        per_sample_weights = None
    
    num_embeddings = input_dict["num_embeddings"]
    embedding_dim = input_dict["embedding_dim"]
    max_norm = input_dict.get("max_norm", None)
    norm_type = input_dict.get("norm_type", 2.0)
    scale_grad_by_freq = input_dict.get("scale_grad_by_freq", False)
    mode = input_dict.get("mode", "mean")
    sparse = input_dict.get("sparse", False)
    include_last_offset = input_dict.get("include_last_offset", False)
    padding_idx = input_dict.get("padding_idx", None)
    
    embeddingbag = torch.nn.EmbeddingBag(num_embeddings, embedding_dim, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, mode=mode, sparse=sparse, include_last_offset=include_last_offset, padding_idx=padding_idx)

    if not cpu:
        input_tensor = input_tensor.cuda()
        if offsets is not None:
            offsets = offsets.cuda()
        if per_sample_weights is not None:
            per_sample_weights = per_sample_weights.cuda()
        embeddingbag = embeddingbag.cuda()
        
    result = embeddingbag(input_tensor, offsets, per_sample_weights)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().cpu().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.int32)
        if "offsets" in input_dict:
            offsets = tf.constant(input_dict["offsets"], dtype=tf.int32)
        else:
            offsets = None
        if "per_sample_weights" in input_dict:
            per_sample_weights = tf.constant(input_dict["per_sample_weights"], dtype=tf.float32)
        else:
            per_sample_weights = None
        
        num_embeddings = input_dict["num_embeddings"]
        embedding_dim = input_dict["embedding_dim"]
        mode = input_dict.get("mode", "mean")
        padding_idx = input_dict.get("padding_idx", None)
        
        embedding_matrix = tf.Variable(tf.random.normal((num_embeddings, embedding_dim)), trainable=False, dtype=tf.float32)
        
        if padding_idx is not None:
            padding_vector = tf.zeros((1, embedding_dim), dtype=tf.float32)
            embedding_matrix = tf.concat([embedding_matrix[:padding_idx], padding_vector, embedding_matrix[padding_idx+1:]], axis=0)

        if input_tensor.shape.rank == 1:
            if offsets is None:
                raise ValueError("Offsets must be provided for 1D input")
            
            embeddings = tf.gather(embedding_matrix, input_tensor)
            
            num_bags = offsets.shape[0]
            
            result_list = []
            for i in range(num_bags):
                start = offsets[i]
                if i < num_bags - 1:
                    end = offsets[i+1]
                else:
                    end = input_tensor.shape[0]
                
                bag_embeddings = embeddings[start:end]
                
                if per_sample_weights is not None:
                    bag_weights = per_sample_weights[start:end]
                    
                    if mode == "sum":
                        weighted_embeddings = bag_embeddings * tf.expand_dims(bag_weights, axis=1)
                        bag_result = tf.reduce_sum(weighted_embeddings, axis=0)
                    else:
                        raise ValueError("per_sample_weights only supported for mode='sum'")
                else:
                    if mode == "sum":
                        bag_result = tf.reduce_sum(bag_embeddings, axis=0)
                    elif mode == "mean":
                        bag_result = tf.reduce_mean(bag_embeddings, axis=0)
                    elif mode == "max":
                        bag_result = tf.reduce_max(bag_embeddings, axis=0)
                    else:
                        raise ValueError("Invalid mode: {}".format(mode))

                result_list.append(bag_result)
            
            result = tf.stack(result_list)
        else:
            embeddings = tf.gather(embedding_matrix, input_tensor)
            
            if mode == "sum":
                result = tf.reduce_sum(embeddings, axis=1)
            elif mode == "mean":
                result = tf.reduce_mean(embeddings, axis=1)
            elif mode == "max":
                result = tf.reduce_max(embeddings, axis=1)
            else:
                raise ValueError("Invalid mode: {}".format(mode))
                
        result = tf.cast(result, tf.float64).numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1, 2, 4, 5, 4, 3, 2, 9], dtype=np.int64),
        "offsets": np.array([0, 4], dtype=np.int64),
        "num_embeddings": 10,
        "embedding_dim": 3,
        "mode": "sum"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([1, 2, 4, 5, 4, 3, 2, 9], dtype=np.int64),
        "offsets": np.array([0, 4], dtype=np.int64),
        "num_embeddings": 10,
        "embedding_dim": 3,
        "mode": "mean"
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[1, 2, 4], [5, 4, 3], [2, 9, 1]], dtype=np.int64),
        "num_embeddings": 10,
        "embedding_dim": 3,
        "mode": "sum"
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([1, 2, 4, 5, 4, 3, 2, 9], dtype=np.int64),
        "offsets": np.array([0, 4], dtype=np.int64),
        "num_embeddings": 10,
        "embedding_dim": 3,
        "mode": "sum",
        "per_sample_weights": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()