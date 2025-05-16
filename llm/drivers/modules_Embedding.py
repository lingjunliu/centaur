import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    num_embeddings = input_dict["num_embeddings"]
    embedding_dim = input_dict["embedding_dim"]
    padding_idx = input_dict.get("padding_idx", None)
    max_norm = input_dict.get("max_norm", None)
    norm_type = input_dict.get("norm_type", 2.0)
    scale_grad_by_freq = input_dict.get("scale_grad_by_freq", False)
    sparse = input_dict.get("sparse", False)
    weight = input_dict.get("weight", None)
    input_tensor = torch.tensor(input_dict["input"])

    if weight is not None:
      weight = torch.tensor(weight)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        if weight is not None:
            weight = weight.cuda()
    
    if weight is None:
        embedding = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse)
    else:
        embedding = torch.nn.Embedding.from_pretrained(weight, freeze=False, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse)
    
    if not cpu:
        embedding = embedding.cuda()

    result = embedding(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    num_embeddings = input_dict["num_embeddings"]
    embedding_dim = input_dict["embedding_dim"]
    padding_idx = input_dict.get("padding_idx", None)
    max_norm = input_dict.get("max_norm", None)
    norm_type = input_dict.get("norm_type", 2.0)
    scale_grad_by_freq = input_dict.get("scale_grad_by_freq", False)
    sparse = input_dict.get("sparse", False)
    weight = input_dict.get("weight", None)
    input_tensor = tf.constant(input_dict["input"])

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        if weight is None:
            embedding = tf.keras.layers.Embedding(num_embeddings, embedding_dim, embeddings_initializer='uniform')
            embedding.build(input_shape=(None,))
        else:
            weight = tf.constant(weight)
            embedding = tf.keras.layers.Embedding(num_embeddings, embedding_dim, weights=[weight], trainable=False)
        
        result = embedding(input_tensor)
        
        if padding_idx is not None:
            mask = tf.cast(tf.equal(input_tensor, padding_idx), tf.float32)
            result = result * (1 - tf.expand_dims(mask, axis=-1))
        
        if max_norm is not None:
            embeddings_norm = tf.norm(result, ord=norm_type, axis=-1, keepdims=True)
            
            def exceed_max_norm_op(x):
                norm = tf.norm(x, ord=norm_type, axis=-1, keepdims=True)
                return x * max_norm / norm
            
            result = tf.where(embeddings_norm > max_norm, exceed_max_norm_op(result), result)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "num_embeddings": 10,
        "embedding_dim": 5,
        "input": np.array([1, 2, 3], dtype=np.int64)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "num_embeddings": 5,
        "embedding_dim": 3,
        "input": np.array([0, 1, 2, 3, 4], dtype=np.int64),
        "weight": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "num_embeddings": 5,
        "embedding_dim": 3,
        "input": np.array([0, 1, 2, 3, 4], dtype=np.int64),
        "weight": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]], dtype=np.float32),
        "padding_idx": 0,
        "max_norm": 1.0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()