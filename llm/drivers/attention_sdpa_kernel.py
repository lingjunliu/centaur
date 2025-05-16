import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from torch.nn.functional import scaled_dot_product_attention

    q = torch.tensor(input_dict["q"])
    k = torch.tensor(input_dict["k"])
    v = torch.tensor(input_dict["v"])
    attn_mask = torch.tensor(input_dict["attn_mask"]) if "attn_mask" in input_dict else None
    dropout_p = input_dict.get("dropout_p", 0.0)
    is_causal = input_dict.get("is_causal", False)

    if not cpu:
        q = q.cuda()
        k = k.cuda()
        v = v.cuda()
        if attn_mask is not None:
            attn_mask = attn_mask.cuda()

    result = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        q = tf.constant(input_dict["q"])
        k = tf.constant(input_dict["k"])
        v = tf.constant(input_dict["v"])
        attn_mask = tf.constant(input_dict["attn_mask"]) if "attn_mask" in input_dict else None
        dropout_p = input_dict.get("dropout_p", 0.0)
        is_causal = input_dict.get("is_causal", False)

        d_k = tf.cast(tf.shape(k)[-1], tf.float32)
        attn_logits = tf.matmul(q, tf.transpose(k, perm=[0, 2, 1])) / tf.math.sqrt(d_k)

        if attn_mask is not None:
            attn_logits = attn_logits + tf.cast(attn_mask, dtype=attn_logits.dtype)

        if is_causal:
            seq_len_q = tf.shape(q)[1]
            seq_len_k = tf.shape(k)[1]
            causal_mask = tf.linalg.band_part(tf.ones((seq_len_q, seq_len_k), dtype=attn_logits.dtype), -1, 0) * -1e9
            attn_logits += causal_mask

        attention_weights = tf.nn.softmax(attn_logits, axis=-1)
        attention_weights = tf.nn.dropout(attention_weights, rate=dropout_p)

        result = tf.matmul(attention_weights, v)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "q": np.random.rand(1, 4, 8).astype(np.float32),
        "k": np.random.rand(1, 4, 8).astype(np.float32),
        "v": np.random.rand(1, 4, 8).astype(np.float32),
        "attn_mask": (np.random.rand(1, 4, 4).astype(np.float32) - 0.5) * 10,
        "dropout_p": 0.0,
        "is_causal": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()