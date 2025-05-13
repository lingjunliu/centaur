import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn as nn

    embedding_dim = input_dict["embedding_dim"]
    num_embeddings = input_dict["num_embeddings"]
    weight = input_dict.get("weight", None)

    input_tensor = torch.tensor(input_dict["input"])

    if not cpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if weight is not None:
        weight = torch.tensor(weight).to(device)
    
    m = nn.Embedding(num_embeddings, embedding_dim)
    if weight is not None:
        m.weight = nn.Parameter(weight)

    m = m.to(device)
    input_tensor = input_tensor.to(device)
    
    result = m(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    embedding_dim = input_dict["embedding_dim"]
    num_embeddings = input_dict["num_embeddings"]
    weight = input_dict.get("weight", None)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):

        if weight is not None:
            weight = tf.constant(weight)
        else:
            weight = tf.random.uniform(shape=[num_embeddings, embedding_dim], minval=-1, maxval=1)
        
        input_tensor = tf.constant(input_dict["input"])
        
        result = tf.nn.embedding_lookup(weight, input_tensor)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1, 0, 2, 0], dtype=np.int64),
        "num_embeddings": 5,
        "embedding_dim": 3,
        "weight": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()