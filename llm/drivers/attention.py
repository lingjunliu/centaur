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
    mask = input_dict.get("mask", None)

    if not cpu:
        query = query.cuda()
        key = key.cuda()
        value = value.cuda()
        if mask is not None:
            mask = torch.tensor(mask).cuda()
    else:
        if mask is not None:
            mask = torch.tensor(mask)

    result = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=dropout_p, is_causal=is_causal)

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
    mask = input_dict.get("mask", None)

    def scaled_dot_product_attention(q, k, v, mask, dropout_rate, is_causal):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits = tf.where(tf.math.is_inf(tf.abs(mask)), mask, scaled_attention_logits)

        if is_causal:
            seq_len = tf.shape(q)[1]
            causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)) * -np.inf, -1, 0)
            scaled_attention_logits = tf.where(tf.math.is_inf(tf.abs(causal_mask)), causal_mask, scaled_attention_logits)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = tf.nn.dropout(attention_weights, rate=dropout_rate)

        output = tf.matmul(attention_weights, v)
        return output

    if not cpu:
        with tf.device("/GPU:0"):
            result = scaled_dot_product_attention(query, key, value, mask, dropout_p, is_causal)
            result = result.numpy()
    else:
        with tf.device("/CPU:0"):
            result = scaled_dot_product_attention(query, key, value, mask, dropout_p, is_causal)
            result = result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "query": np.random.rand(2, 3, 4).astype(np.float32),
        "key": np.random.rand(2, 4, 4).astype(np.float32),
        "value": np.random.rand(2, 4, 5).astype(np.float32),
        "dropout_p": 0.1,
        "is_causal": False,
        "mask": (np.random.randint(0, 2, size=(2, 3, 4)).astype(np.float32) -1)* np.inf
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL, equal_nan=True), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()