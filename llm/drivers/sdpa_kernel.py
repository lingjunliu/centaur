import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn.functional as F

    q = torch.tensor(input_dict["q"])
    k = torch.tensor(input_dict["k"])
    v = torch.tensor(input_dict["v"])
    mask = input_dict.get("mask", None)
    if mask is not None:
        mask = torch.tensor(input_dict["mask"])
    scale = input_dict.get("scale", None)
    dropout_p = input_dict.get("dropout_p", 0.0)
    is_causal = input_dict.get("is_causal", False)
    training = input_dict.get("training", False)

    if not cpu:
        q = q.cuda()
        k = k.cuda()
        v = v.cuda()
        if mask is not None:
            mask = mask.cuda()

    result = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=is_causal)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    q = tf.constant(input_dict["q"])
    k = tf.constant(input_dict["k"])
    v = tf.constant(input_dict["v"])
    mask = input_dict.get("mask", None)
    scale = input_dict.get("scale", None)
    dropout_p = input_dict.get("dropout_p", 0.0)
    is_causal = input_dict.get("is_causal", False)
    training = input_dict.get("training", False)

    if mask is not None:
        mask = tf.constant(input_dict["mask"])
        bool_mask = tf.cast(mask, tf.bool)
        mask = tf.where(bool_mask, 0.0, tf.float32.min)

    if scale is None:
        scale = tf.cast(tf.shape(k)[-1], tf.float32) ** -0.5
    else:
        scale = tf.constant(scale, dtype=tf.float32)

    attn_logits = tf.matmul(q, k, transpose_b=True) * scale

    if mask is not None:
        attn_logits += mask

    if is_causal:
        seq_len = tf.shape(q)[-2]
        causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0) * -1e9
        attn_logits += causal_mask

    attn_weights = tf.nn.softmax(attn_logits, axis=-1)
    
    if dropout_p > 0 and training:
        attn_weights = tf.nn.dropout(attn_weights, rate=dropout_p)
    
    result = tf.matmul(attn_weights, v)

    return {"result": result.numpy()}


def main():
    A_TOL = 0.01

    input_data = {
        "q": np.random.rand(2, 4, 8).astype(np.float32),
        "k": np.random.rand(2, 5, 8).astype(np.float32),
        "v": np.random.rand(2, 5, 8).astype(np.float32),
        "dropout_p": 0.1,
        "is_causal": False,
        "mask": np.random.randint(0, 2, size=(2, 4, 5)).astype(np.bool_)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()