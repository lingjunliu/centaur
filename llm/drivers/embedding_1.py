import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    num_embeddings = input_dict.get("num_embeddings")
    embedding_dim = input_dict.get("embedding_dim")
    padding_idx = input_dict.get("padding_idx")
    max_norm = input_dict.get("max_norm")
    norm_type = input_dict.get("norm_type", 2.0)
    scale_grad_by_freq = input_dict.get("scale_grad_by_freq", False)
    sparse = input_dict.get("sparse", False)
    weight = input_dict.get("weight")

    if weight is not None:
        weight = torch.tensor(weight)
    else:
        weight = None

    if not cpu:
        input_tensor = input_tensor.cuda()
        if weight is not None:
            weight = weight.cuda()

    if weight is None:
        embedding = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse)
        if not cpu:
            embedding = embedding.cuda()
        result = embedding(input_tensor)
    else:
        embedding = torch.nn.Embedding.from_pretrained(weight, freeze=False, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse)
        if not cpu:
            embedding = embedding.cuda()
        result = embedding(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = input_dict["input"]
    num_embeddings = input_dict.get("num_embeddings")
    embedding_dim = input_dict.get("embedding_dim")
    padding_idx = input_dict.get("padding_idx")
    max_norm = input_dict.get("max_norm")
    norm_type = input_dict.get("norm_type", 2.0)
    scale_grad_by_freq = input_dict.get("scale_grad_by_freq", False)
    sparse = input_dict.get("sparse", False)
    weight = input_dict.get("weight")
    
    if weight is not None:
        weight = weight
    else:
        weight = None

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        if weight is None:
            embedding = tf.keras.layers.Embedding(num_embeddings, embedding_dim, embeddings_initializer='uniform', embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, mask_zero=False, input_length=None)
            embedding_matrix = embedding.weights[0]
            if padding_idx is not None:
                mask = tf.one_hot([padding_idx], depth=num_embeddings)
                embedding_matrix = tf.where(mask, tf.zeros(embedding_dim, dtype=tf.float32), embedding_matrix)
            result = tf.nn.embedding_lookup(embedding_matrix, input_tensor, max_norm=max_norm)
        else:
            embedding_matrix = tf.constant(weight, dtype=tf.float32)
            result = tf.nn.embedding_lookup(embedding_matrix, input_tensor, max_norm=max_norm)
        if max_norm is not None:
            embeddings_norm = tf.norm(result, axis=1, keepdims=True)
            desired_norm = tf.clip_by_value(embeddings_norm, 0, max_norm)
            result = result * (desired_norm / (embeddings_norm + 1e-8))
    
    return {"result": result.numpy()}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1, 0, 2, 0], dtype=np.int64),
        "num_embeddings": 5,
        "embedding_dim": 3,
        "padding_idx": 0,
        "weight": np.array([[0.1, 0.2, 0.3],
                           [0.4, 0.5, 0.6],
                           [0.7, 0.8, 0.9],
                           [1.0, 1.1, 1.2],
                           [1.3, 1.4, 1.5]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()