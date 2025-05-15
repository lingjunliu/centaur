import numpy as np
import tensorflow as tf

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    offsets = torch.tensor(input_dict["offsets"])
    indices = torch.tensor(input_dict["indices"])
    sparse = input_dict.get("sparse", False)
    mode = input_dict.get("mode", "mean")
    per_sample_weights = torch.tensor(input_dict["per_sample_weights"]) if "per_sample_weights" in input_dict else None
    include_last_offset = input_dict.get("include_last_offset", False)
    padding_idx = input_dict.get("padding_idx", None)

    if not cpu:
        input_tensor = input_tensor.cuda()
        offsets = offsets.cuda()
        indices = indices.cuda()
        if per_sample_weights is not None:
            per_sample_weights = per_sample_weights.cuda()

    emb = torch.nn.EmbeddingBag(input_tensor.shape[1], 5, mode="sum", sparse=sparse, include_last_offset=include_last_offset, padding_idx=padding_idx)
    emb.weight = torch.nn.Parameter(input_tensor)

    result = emb(indices, offsets, per_sample_weights=per_sample_weights)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        offsets = tf.constant(input_dict["offsets"], dtype=tf.int32)
        indices = tf.constant(input_dict["indices"])
        mode = input_dict.get("mode", "mean")
        per_sample_weights = tf.constant(input_dict["per_sample_weights"]) if "per_sample_weights" in input_dict else None
        include_last_offset = input_dict.get("include_last_offset", False)
        padding_idx = input_dict.get("padding_idx", None)
        
        num_embeddings = tf.shape(input_tensor)[0]
        embedding_dim = tf.shape(input_tensor)[1]

        def embedding_bag(indices, offsets, weights, mode):
            num_segments = tf.shape(offsets)[0]
            segment_ids = []
            
            
            for i in range(num_segments):
                start = offsets[i].numpy()
                if i + 1 < num_segments:
                    end = offsets[i+1].numpy()
                else:
                    end = tf.shape(indices)[0].numpy()
                length = end - start
                segment_ids.extend([i] * length)
            
            segment_ids = tf.constant(segment_ids, dtype=tf.int32)
            
            if weights is not None:
                embeddings = tf.gather(input_tensor, indices) * weights
            else:
                embeddings = tf.gather(input_tensor, indices)
            
            
            result = tf.math.segment_sum(embeddings, segment_ids)
            
            return result
            
        result = embedding_bag(indices, offsets, per_sample_weights, "sum")

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
                           [0.6, 0.7, 0.8, 0.9, 1.0],
                           [1.1, 1.2, 1.3, 1.4, 1.5],
                           [1.6, 1.7, 1.8, 1.9, 2.0]], dtype=np.float32),
        "offsets": np.array([0, 1, 2], dtype=np.int64),
        "indices": np.array([0, 1, 2, 3], dtype=np.int64),
        "per_sample_weights": np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        "sparse": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()