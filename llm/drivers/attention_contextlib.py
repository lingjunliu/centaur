import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from torch.nn.functional import scaled_dot_product_attention

    query = torch.tensor(input_dict["query"])
    key = torch.tensor(input_dict["key"])
    value = torch.tensor(input_dict["value"])
    attn_mask = torch.tensor(input_dict["attn_mask"]) if "attn_mask" in input_dict else None
    dropout_p = input_dict.get("dropout_p", 0.0)
    is_causal = input_dict.get("is_causal", False)

    if not cpu:
        query = query.cuda()
        key = key.cuda()
        value = value.cuda()
        if attn_mask is not None:
            attn_mask = attn_mask.cuda()

    result = scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        query = tf.constant(input_dict["query"])
        key = tf.constant(input_dict["key"])
        value = tf.constant(input_dict["value"])
        attn_mask = tf.constant(input_dict["attn_mask"]) if "attn_mask" in input_dict else None
        dropout_p = input_dict.get("dropout_p", 0.0)
        is_causal = input_dict.get("is_causal", False)

        d_k = tf.cast(tf.shape(key)[-1], tf.float32)
        scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(d_k)

        if attn_mask is not None:
            scores = scores + attn_mask

        if is_causal:
            seq_len = tf.shape(query)[-2]
            causal_mask = tf.experimental.numpy.triu(tf.ones((seq_len, seq_len)) * -1e10, k=1)
            scores = scores + causal_mask

        attention_weights = tf.nn.softmax(scores, axis=-1)
        attention_weights = tf.nn.dropout(attention_weights, rate=dropout_p)

        result = tf.matmul(attention_weights, value)

    return {"result": result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "query": np.random.rand(2, 3, 4).astype(np.float32),
        "key": np.random.rand(2, 3, 4).astype(np.float32),
        "value": np.random.rand(2, 3, 4).astype(np.float32),
        "attn_mask": np.random.rand(2, 3, 3).astype(np.float32) - 0.5 # Adding negative values to test masking
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()