import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    from torch.nn.functional import scaled_dot_product_attention

    q = torch.tensor(input_dict["q"])
    k = torch.tensor(input_dict["k"])
    v = torch.tensor(input_dict["v"])
    attn_mask = input_dict.get("attn_mask", None)
    dropout_p = input_dict.get("dropout_p", 0.0)
    is_causal = input_dict.get("is_causal", False)
    scale = input_dict.get("scale", None)

    if not cpu:
        q = q.cuda()
        k = k.cuda()
        v = v.cuda()
        if attn_mask is not None:
            attn_mask = torch.tensor(attn_mask, dtype=torch.bool).cuda()
    else:
        if attn_mask is not None:
            attn_mask = torch.tensor(attn_mask, dtype=torch.bool)
    
    result = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
    
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
        q = tf.constant(input_dict["q"])
        k = tf.constant(input_dict["k"])
        v = tf.constant(input_dict["v"])
        attn_mask = input_dict.get("attn_mask", None)
        dropout_p = input_dict.get("dropout_p", 0.0)
        is_causal = input_dict.get("is_causal", False)
        scale = input_dict.get("scale", None)

        if scale is None:
            d_k = tf.cast(tf.shape(k)[-1], tf.float32)
            scale = tf.math.rsqrt(d_k)

        matmul_qk = tf.matmul(q, k, transpose_b=True)
        scaled_attention_logits = matmul_qk * scale

        if attn_mask is not None:
            scaled_attention_logits = tf.where(tf.cast(attn_mask, tf.bool), scaled_attention_logits, tf.constant(-1e9, dtype=q.dtype))
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = tf.nn.dropout(attention_weights, rate=dropout_p)

        output = tf.matmul(attention_weights, v)
        result = output.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "q": np.random.rand(2, 3, 4).astype(np.float32),
        "k": np.random.rand(2, 3, 4).astype(np.float32),
        "v": np.random.rand(2, 3, 4).astype(np.float32),
        "attn_mask": np.random.randint(0, 2, size=(2, 3, 3)).astype(np.bool_),
        "dropout_p": 0.1,
        "is_causal": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()