import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    embed_dim = input_dict["embed_dim"]
    num_heads = input_dict["num_heads"]
    bias = input_dict.get("bias", True)
    add_bias_kv = input_dict.get("add_bias_kv", False)
    add_zero_attn = input_dict.get("add_zero_attn", False)
    dropout = input_dict.get("dropout", 0.0)
    batch_first = input_dict.get("batch_first", False)
    kdim = input_dict.get("kdim", None)
    vdim = input_dict.get("vdim", None)

    mha = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, bias=bias, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn, batch_first=batch_first, kdim=kdim, vdim=vdim)

    query = torch.tensor(input_dict["query"], requires_grad=False)
    key = torch.tensor(input_dict["key"], requires_grad=False)
    value = torch.tensor(input_dict["value"], requires_grad=False)
    
    if not cpu:
        mha = mha.cuda()
        query = query.cuda()
        key = key.cuda()
        value = value.cuda()
    
    attn_output, attn_output_weights = mha(query, key, value)
    
    if not cpu:
        attn_output = attn_output.cpu()
        attn_output_weights = attn_output_weights.cpu()
    
    return {"attn_output": attn_output.detach().numpy(), "attn_output_weights": attn_output_weights.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    embed_dim = input_dict["embed_dim"]
    num_heads = input_dict["num_heads"]
    bias = input_dict.get("bias", True)
    add_bias_kv = input_dict.get("add_bias_kv", False)
    add_zero_attn = input_dict.get("add_zero_attn", False)
    dropout = input_dict.get("dropout", 0.0)
    batch_first = input_dict.get("batch_first", False)
    kdim = input_dict.get("kdim", embed_dim)
    vdim = input_dict.get("vdim", embed_dim)
    

    query = tf.constant(input_dict["query"])
    key = tf.constant(input_dict["key"])
    value = tf.constant(input_dict["value"])

    def scaled_dot_product_attention(q, k, v, mask=None, dropout=0.0):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = tf.nn.dropout(attention_weights, rate=dropout)
        output = tf.matmul(attention_weights, v)

        return output, attention_weights
    
    def split_heads(x, num_heads, embed_dim):
        if batch_first:
            x = tf.transpose(x, perm=[1, 0, 2])
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, num_heads, embed_dim // num_heads))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def combine_heads(x, num_heads, embed_dim):
        batch_size = tf.shape(x)[0]
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (batch_size, -1, embed_dim))
        if batch_first:
            x = tf.transpose(x, perm=[1, 0, 2])
        return x
    
    Wq = tf.Variable(tf.random.normal(shape=(embed_dim, embed_dim)))
    Wk = tf.Variable(tf.random.normal(shape=(kdim, embed_dim)))
    Wv = tf.Variable(tf.random.normal(shape=(vdim, embed_dim)))
    Wo = tf.Variable(tf.random.normal(shape=(embed_dim, embed_dim)))

    bq = tf.Variable(tf.zeros(embed_dim))
    bk = tf.Variable(tf.zeros(embed_dim))
    bv = tf.Variable(tf.zeros(embed_dim))
    bo = tf.Variable(tf.zeros(embed_dim))

    q = split_heads(tf.matmul(query, Wq) + (bq if bias else 0.0), num_heads, embed_dim)
    k = split_heads(tf.matmul(key, Wk) + (bk if bias else 0.0), num_heads, embed_dim)
    v = split_heads(tf.matmul(value, Wv) + (bv if bias else 0.0), num_heads, embed_dim)

    attn_output, attn_output_weights = scaled_dot_product_attention(q, k, v, dropout=dropout)

    attn_output = combine_heads(attn_output, num_heads, embed_dim)
    attn_output = tf.matmul(attn_output, Wo) + (bo if bias else 0.0)

    return {"attn_output": attn_output.numpy(), "attn_output_weights": attn_output_weights.numpy()}

def main():
    A_TOL = 0.1
    input_data = {
        "query": np.random.rand(2, 4, 8).astype(np.float32),
        "key": np.random.rand(2, 4, 8).astype(np.float32),
        "value": np.random.rand(2, 4, 8).astype(np.float32),
        "embed_dim": 8,
        "num_heads": 4,
        "dropout": 0.1,
        "batch_first": False,
        "kdim": 8,
        "vdim": 8
    }

    torch_result = torch_version(input_data)
    
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["attn_output"], tf_result["attn_output"], atol=A_TOL), "Results do not match"
    assert np.allclose(torch_result["attn_output_weights"], tf_result["attn_output_weights"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()