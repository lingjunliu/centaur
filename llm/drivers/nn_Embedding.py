import numpy as np
import tensorflow as tf

def torch_version(input_dict, cpu=True):
    import torch

    num_embeddings = input_dict["num_embeddings"]
    embedding_dim = input_dict["embedding_dim"]
    input_tensor = torch.tensor(input_dict["input"])
    padding_idx = input_dict.get("padding_idx", None)
    max_norm = input_dict.get("max_norm", None)
    norm_type = input_dict.get("norm_type", 2.0)
    scale_grad_by_freq = input_dict.get("scale_grad_by_freq", False)
    sparse = input_dict.get("sparse", False)

    embedding = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse)
    
    if not cpu:
        embedding = embedding.cuda()
        input_tensor = input_tensor.cuda()

    result = embedding(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    num_embeddings = input_dict["num_embeddings"]
    embedding_dim = input_dict["embedding_dim"]
    input_tensor = tf.constant(input_dict["input"])
    padding_idx = input_dict.get("padding_idx", None)
    max_norm = input_dict.get("max_norm", None)
    norm_type = input_dict.get("norm_type", 2.0)
    scale_grad_by_freq = input_dict.get("scale_grad_by_freq", False)
    sparse = input_dict.get("sparse", False)
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        embedding = tf.keras.layers.Embedding(num_embeddings, embedding_dim, embeddings_initializer='normal', mask_zero=(padding_idx is not None))
        
        result = embedding(input_tensor)
            
        if max_norm is not None:
            embeddings = embedding.trainable_variables[0]

            def normalize(x):
                l2_norm = tf.norm(x, ord=norm_type)
                
                is_larger = tf.greater(l2_norm, max_norm)
                
                factor = tf.where(is_larger, max_norm / l2_norm, tf.constant(1.0, dtype=x.dtype))
                
                return x * factor
            
            normalized_embeddings = tf.vectorized_map(normalize, embeddings)
            embedding.set_weights([normalized_embeddings])
            result = embedding(input_tensor)
        if padding_idx is not None:
            mask = tf.cast(tf.not_equal(input_tensor, padding_idx), dtype=tf.float32)
            mask = tf.expand_dims(mask, axis=-1)
            result = result * mask

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "num_embeddings": 10,
        "embedding_dim": 3,
        "input": np.array([1, 2, 0, 5], dtype=np.int32),
        "padding_idx": 0,
        "max_norm": 1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()