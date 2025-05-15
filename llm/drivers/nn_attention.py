import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn.functional as F

    query = torch.tensor(input_dict["query"])
    key = torch.tensor(input_dict["key"])
    value = torch.tensor(input_dict["value"])
    embed_dim_to_check = input_dict.get("embed_dim_to_check", None)
    dropout_p = input_dict.get("dropout_p", 0.0)
    is_causal = input_dict.get("is_causal", False)
    bias = input_dict.get("bias", True)
    add_zero_attn = input_dict.get("add_zero_attn", False)

    if not cpu:
        query = query.cuda()
        key = key.cuda()
        value = value.cuda()

    result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=dropout_p, is_causal=is_causal)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    query = tf.constant(input_dict["query"])
    key = tf.constant(input_dict["key"])
    value = tf.constant(input_dict["value"])
    dropout_p = input_dict.get("dropout_p", 0.0)
    is_causal = input_dict.get("is_causal", False)

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # Calculate attention weights
        d_k = tf.cast(tf.shape(key)[-1], dtype=tf.float32)
        attention_logits = tf.matmul(query, key, transpose_b=True)
        attention_logits = attention_logits / tf.math.sqrt(d_k)
        
        if is_causal:
            seq_len = tf.shape(query)[1]
            batch_size = tf.shape(query)[0]
            causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
            causal_mask = tf.reshape(causal_mask, (1, seq_len, seq_len))
            causal_mask = tf.tile(causal_mask, [batch_size, 1, 1])
            attention_logits = tf.where(causal_mask == 0, tf.float32.min, attention_logits)
        
        attention_weights = tf.nn.softmax(attention_logits, axis=-1)
        
        if dropout_p > 0.0:
            attention_weights = tf.nn.dropout(attention_weights, rate=dropout_p)

        # Calculate context vector
        result = tf.matmul(attention_weights, value)
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "query": np.random.rand(2, 3, 4).astype(np.float32),
        "key": np.random.rand(2, 3, 4).astype(np.float32),
        "value": np.random.rand(2, 3, 5).astype(np.float32),
        "dropout_p": 0.1,
        "is_causal": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()