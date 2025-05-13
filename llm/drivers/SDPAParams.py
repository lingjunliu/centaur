import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    import torch.nn.functional as F

    q = torch.tensor(input_dict["q"])
    k = torch.tensor(input_dict["k"])
    v = torch.tensor(input_dict["v"])
    mask = torch.tensor(input_dict["mask"]) if "mask" in input_dict else None
    scale = input_dict.get("scale", None)
    dropout_p = input_dict.get("dropout_p", 0.0)
    is_causal = input_dict.get("is_causal", False)
    
    if not cpu:
        q = q.cuda()
        k = k.cuda()
        v = v.cuda()
        if mask is not None:
            mask = mask.cuda()

    attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=is_causal)
    
    if not cpu:
        attn_output = attn_output.cpu()
    
    return {"result": attn_output.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        q = tf.constant(input_dict["q"])
        k = tf.constant(input_dict["k"])
        v = tf.constant(input_dict["v"])
        mask = tf.constant(input_dict["mask"]) if "mask" in input_dict else None
        scale = input_dict.get("scale", None)
        dropout_p = input_dict.get("dropout_p", 0.0)
        is_causal = input_dict.get("is_causal", False)

        if scale is None:
            scale = tf.cast(tf.shape(k)[-1], tf.float32) ** -0.5
        
        attn_weights = tf.matmul(q, k, transpose_b=True) * scale
        
        if mask is not None:
            attn_weights = tf.where(mask, -float('inf'), attn_weights)

        if is_causal:
            seq_len = tf.shape(q)[-2]
            causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=tf.int32), -1, 0)
            causal_mask = tf.cast(causal_mask, dtype=tf.bool)
            attn_weights = tf.where(tf.logical_not(causal_mask), tf.constant(-np.inf, dtype=tf.float32), attn_weights)

        attn_weights = tf.nn.softmax(attn_weights)
        
        attn_weights = tf.nn.dropout(attn_weights, rate=dropout_p)

        attn_output = tf.matmul(attn_weights, v)
        
        attn_output = attn_output.numpy()
    
    return {"result": attn_output}

def main():
    A_TOL = 0.01

    input_data = {
        "q": np.random.rand(2, 3, 4).astype(np.float32),
        "k": np.random.rand(2, 3, 4).astype(np.float32),
        "v": np.random.rand(2, 3, 4).astype(np.float32),
        "dropout_p": 0.1,
        "is_causal": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()