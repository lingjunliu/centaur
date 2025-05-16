import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn.functional as F

    input_tensor = torch.tensor(input_dict["query"])
    key = torch.tensor(input_dict["key"])
    value = torch.tensor(input_dict["value"])
    need_weights = input_dict.get("need_weights", True)
    attn_mask = input_dict.get("attn_mask", None)
    if attn_mask is not None:
        attn_mask = torch.tensor(attn_mask)

    if not cpu:
        input_tensor = input_tensor.cuda()
        key = key.cuda()
        value = value.cuda()
        if attn_mask is not None:
            attn_mask = attn_mask.cuda()

    try:
        result, weights = F.scaled_dot_product_attention(input_tensor, key, value, attn_mask=attn_mask, is_causal=False, scale=None)
    except ValueError:
        result = F.scaled_dot_product_attention(input_tensor, key, value, attn_mask=attn_mask, is_causal=False, scale=None)
        weights = None

    if not cpu:
        result = result.cpu()
        if weights is not None:
            weights = weights.cpu()

    if weights is not None:
        return {"result": result.numpy(), "weights": weights.numpy()}
    else:
        return {"result": result.numpy(), "weights": None}

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
        need_weights = input_dict.get("need_weights", True)
        attn_mask = input_dict.get("attn_mask", None)
        if attn_mask is not None:
            attn_mask = tf.constant(attn_mask)

        def scaled_dot_product_attention(q, k, v, mask):
            d_k = tf.cast(tf.shape(k)[-1], tf.float32)
            matmul_qk = tf.matmul(q, k, transpose_b=True)
            scaled_attention_logits = matmul_qk / tf.math.sqrt(d_k)

            if mask is not None:
                scaled_attention_logits += (mask * -1e9)

            attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
            output = tf.matmul(attention_weights, v)
            if need_weights:
                return output, attention_weights
            else:
                return output, None

        result, weights = scaled_dot_product_attention(query, key, value, attn_mask)
        result = result.numpy()
        if weights is not None:
            weights = weights.numpy()
    if weights is not None:
        return {"result": result, "weights": weights}
    else:
        return {"result": result, "weights": None}

def main():
    A_TOL = 0.01
    input_data = {
        "query": np.array([[1, 2, 3]], dtype=np.float32),
        "key": np.array([[4, 5, 6]], dtype=np.float32),
        "value": np.array([[7, 8, 9]], dtype=np.float32),
        "need_weights": True,
        "attn_mask": None
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    if torch_result["weights"] is not None and tf_result["weights"] is not None:
        assert np.allclose(torch_result["weights"], tf_result["weights"], atol=A_TOL), "Weights do not match"

    print("Success")

if __name__ == "__main__":
    main()