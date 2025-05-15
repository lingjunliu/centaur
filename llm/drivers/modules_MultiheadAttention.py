import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    query = torch.tensor(input_dict["query"])
    key = torch.tensor(input_dict["key"])
    value = torch.tensor(input_dict["value"])
    embed_dim_to_check = input_dict["embed_dim_to_check"]
    num_heads = input_dict["num_heads"]
    dropout = input_dict.get("dropout", 0.0)
    bias = input_dict.get("bias", True)
    add_bias_kv = input_dict.get("add_bias_kv", False)
    add_zero_attn = input_dict.get("add_zero_attn", False)
    kdim = input_dict.get("kdim", None)
    vdim = input_dict.get("vdim", None)
    batch_first = input_dict.get("batch_first", False)

    if not cpu:
        query = query.cuda()
        key = key.cuda()
        value = value.cuda()

    multihead_attn = torch.nn.MultiheadAttention(embed_dim_to_check, num_heads, dropout=dropout, bias=bias, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn, kdim=kdim, vdim=vdim, batch_first=batch_first)

    result, attn_output_weights = multihead_attn(query, key, value)
    
    if not cpu:
        result = result.cpu()
        attn_output_weights = attn_output_weights.cpu()
    
    return {"result": result.detach().numpy(), "attn_output_weights": attn_output_weights.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    query = tf.constant(input_dict["query"])
    key = tf.constant(input_dict["key"])
    value = tf.constant(input_dict["value"])
    embed_dim_to_check = input_dict["embed_dim_to_check"]
    num_heads = input_dict["num_heads"]
    dropout = input_dict.get("dropout", 0.0)
    bias = input_dict.get("bias", True)
    add_bias_kv = input_dict.get("add_bias_kv", False)
    add_zero_attn = input_dict.get("add_zero_attn", False)
    kdim = input_dict.get("kdim", None)
    vdim = input_dict.get("vdim", None)
    batch_first = input_dict.get("batch_first", False)

    if batch_first:
        query = tf.transpose(query, perm=[1, 0, 2])
        key = tf.transpose(key, perm=[1, 0, 2])
        value = tf.transpose(value, perm=[1, 0, 2])

    if kdim is None:
        kdim = embed_dim_to_check
    if vdim is None:
        vdim = embed_dim_to_check

    q = tf.keras.layers.Dense(embed_dim_to_check, use_bias=bias, kernel_initializer='glorot_uniform', bias_initializer='zeros')(query)
    k = tf.keras.layers.Dense(embed_dim_to_check, use_bias=bias, kernel_initializer='glorot_uniform', bias_initializer='zeros')(key)
    v = tf.keras.layers.Dense(embed_dim_to_check, use_bias=bias, kernel_initializer='glorot_uniform', bias_initializer='zeros')(value)

    q = tf.reshape(q, shape=(-1, tf.shape(q)[1], num_heads, embed_dim_to_check // num_heads))
    k = tf.reshape(k, shape=(-1, tf.shape(k)[1], num_heads, embed_dim_to_check // num_heads))
    v = tf.reshape(v, shape=(-1, tf.shape(v)[1], num_heads, embed_dim_to_check // num_heads))

    q = tf.transpose(q, perm=[0, 2, 1, 3])  # (batch_size, num_heads, seq_len, depth)
    k = tf.transpose(k, perm=[0, 2, 1, 3])
    v = tf.transpose(v, perm=[0, 2, 1, 3])

    attn_output_weights = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2])) / tf.math.sqrt(tf.cast(embed_dim_to_check // num_heads, dtype=tf.float32))
    
    if add_zero_attn:
        attn_output_weights = tf.pad(attn_output_weights, [[0, 0], [0, 0], [0, 0], [0, 1]])

    attn_output_weights = tf.nn.softmax(attn_output_weights, axis=-1)
    attn_output_weights = tf.nn.dropout(attn_output_weights, rate=dropout)

    attn_output = tf.matmul(attn_output_weights, v)
    attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
    attn_output = tf.reshape(attn_output, shape=(-1, tf.shape(attn_output)[1], embed_dim_to_check))

    result = tf.keras.layers.Dense(embed_dim_to_check, use_bias=bias, kernel_initializer='glorot_uniform', bias_initializer='zeros')(attn_output)

    if batch_first:
        result = tf.transpose(result, perm=[1, 0, 2])
        attn_output_weights = tf.transpose(attn_output_weights, perm=[1, 0, 2, 3])
    
    attn_output_weights = tf.reduce_mean(attn_output_weights, axis=0)
    
    return {"result": result.numpy(), "attn_output_weights": attn_output_weights.numpy()}

def main():
    A_TOL = 0.1

    input_data = {
        "query": np.random.rand(2, 3, 4).astype(np.float32),
        "key": np.random.rand(2, 3, 4).astype(np.float32),
        "value": np.random.rand(2, 3, 4).astype(np.float32),
        "embed_dim_to_check": 4,
        "num_heads": 2,
        "dropout": 0.1,
        "bias": True,
        "add_bias_kv": False,
        "add_zero_attn": False,
        "batch_first": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    assert np.allclose(torch_result["attn_output_weights"], tf_result["attn_output_weights"], atol=A_TOL), "Attention weights do not match"

    print("Success")

if __name__ == "__main__":
    main()