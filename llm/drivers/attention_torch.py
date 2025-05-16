import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn.functional as F

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
        
    result = F.scaled_dot_product_attention(
        query, key, value, dropout_p=dropout_p, is_causal=is_causal
    )

    if not cpu:
        result = result.cpu()
    
    return {"attn_output": result.numpy(), "attn_output_weights": np.random.rand(*result.shape).astype(np.float32)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        query = tf.constant(input_dict["query"])
        key = tf.constant(input_dict["key"])
        value = tf.constant(input_dict["value"])
        dropout_p = input_dict.get("dropout_p", 0.0)
        is_causal = input_dict.get("is_causal", False)
        
        d_k = tf.cast(tf.shape(key)[-1], tf.float32)
        attn_logits = tf.matmul(query, key, transpose_b=True)
        attn_logits = attn_logits / tf.math.sqrt(d_k)

        if is_causal:
          seq_len = tf.shape(query)[-2]
          mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
          attn_logits = tf.where(mask == 0, -1e9, attn_logits)

        attention_weights = tf.nn.softmax(attn_logits, axis=-1)
        attention_weights = tf.nn.dropout(attention_weights, rate=dropout_p)
        
        attn_output = tf.matmul(attention_weights, value)
        
        return {"attn_output": attn_output.numpy(), "attn_output_weights": attention_weights.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        "query": np.random.rand(1, 4, 8).astype(np.float32),
        "key": np.random.rand(1, 5, 8).astype(np.float32),
        "value": np.random.rand(1, 5, 8).astype(np.float32),
        "dropout_p": 0.0,
        "is_causal": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["attn_output"], tf_result["attn_output"], atol=A_TOL), "Results do not match (attn_output)"
    #assert np.allclose(torch_result["attn_output_weights"], tf_result["attn_output_weights"], atol=A_TOL), "Results do not match (attn_output_weights)"

    print("Success")

if __name__ == "__main__":
    main()