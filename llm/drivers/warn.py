import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from torch.nn.functional import scaled_dot_product_attention

    q = torch.tensor(input_dict["q"])
    k = torch.tensor(input_dict["k"])
    v = torch.tensor(input_dict["v"])
    attn_mask = input_dict.get("attn_mask", None)
    if attn_mask is not None:
        attn_mask = torch.tensor(input_dict["attn_mask"])
    dropout_p = input_dict.get("dropout_p", 0.0)
    is_causal = input_dict.get("is_causal", False)

    if not cpu:
        q = q.cuda()
        k = k.cuda()
        v = v.cuda()
        if attn_mask is not None:
            attn_mask = attn_mask.cuda()

    torch.nn.attention.warn = input_dict["warn"]

    result = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}


def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    from tensorflow.keras.layers import Layer, Softmax, Dropout
    
    class ScaledDotProductAttention(Layer):
        def __init__(self, dropout_rate=0.0):
            super().__init__()
            self.softmax = Softmax(axis=-1)
            self.dropout = Dropout(dropout_rate)

        def call(self, q, k, v, mask=None):
            d_k = tf.cast(tf.shape(k)[-1], dtype=tf.float32)
            attention_scores = tf.matmul(q, k, transpose_b=True)
            attention_scores = attention_scores / tf.math.sqrt(d_k)

            if mask is not None:
                attention_scores += (mask * -1e9)

            attention_probs = self.softmax(attention_scores)
            attention_probs = self.dropout(attention_probs)
            output = tf.matmul(attention_probs, v)
            return output

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        q = tf.constant(input_dict["q"])
        k = tf.constant(input_dict["k"])
        v = tf.constant(input_dict["v"])
        attn_mask = input_dict.get("attn_mask", None)
        if attn_mask is not None:
            attn_mask = tf.constant(input_dict["attn_mask"])

        dropout_p = input_dict.get("dropout_p", 0.0)
        is_causal = input_dict.get("is_causal", False)
        
        attention_layer = ScaledDotProductAttention(dropout_rate=dropout_p)
        result = attention_layer(q, k, v, mask=attn_mask)

        result = result.numpy()

    return {"result": result}


def main():
    A_TOL = 0.01

    input_data = {
        "q": np.random.rand(2, 1, 4).astype(np.float32),
        "k": np.random.rand(2, 3, 4).astype(np.float32),
        "v": np.random.rand(2, 3, 4).astype(np.float32),
        "attn_mask": np.random.randint(0, 2, size=(2, 1, 3)).astype(np.float32) - 1,
        "dropout_p": 0.1,
        "is_causal": False,
        "warn": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")


if __name__ == "__main__":
    main()