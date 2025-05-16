import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True.nn.functional as F

    q = torch.tensor(input_dict["q"])
    k = torch.tensor(input_dict["k"])
    v = torch.tensor(input_dict["v"])
    key_padding_mask = torch.tensor(input_dict.get("key_padding_mask", np.zeros((q.shape[0], k.shape[1]), dtype=bool)))
    need_weights = input_dict.get("need_weights", True)
    attn_mask = torch.tensor(input_dict.get("attn_mask", np.zeros((q.shape[0], q.shape[1], k.shape[1]))))
    average_attn_weights = input_dict.get("average_attn_weights", True)
    is_causal = input_dict.get("is_causal", False)

    if not cpu:
        q = q.cuda()
        k = k.cuda()
        v = v.cuda()
        key_padding_mask = key_padding_mask.cuda()
        attn_mask = attn_mask.cuda()

    result = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
    
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
        key_padding_mask = tf.constant(input_dict.get("key_padding_mask", np.zeros((q.shape[0], k.shape[1]), dtype=np.bool_)))
        need_weights = input_dict.get("need_weights", True)
        attn_mask = tf.constant(input_dict.get("attn_mask", np.zeros((q.shape[0], q.shape[1], k.shape[1]))))
        average_attn_weights = input_dict.get("average_attn_weights", True)
        is_causal = input_dict.get("is_causal", False)

        def attention(q, k, v, mask=None, is_causal=False):
            d_k = tf.cast(tf.shape(k)[-1], tf.float32)
            scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(d_k)
            
            if mask is not None:
                scores = scores + mask

            if is_causal:
                seq_len = tf.shape(q)[1]
                causal_mask = tf.cast(tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0), tf.bool)
                scores = tf.where(causal_mask, scores, float('-inf'))

            attention_weights = tf.nn.softmax(scores, axis=-1)
            return tf.matmul(attention_weights, v)

        attn_mask_inf = tf.where(tf.cast(attn_mask, dtype=tf.bool), np.float32(-np.inf), tf.zeros_like(attn_mask, dtype=tf.float32))
        attn_output = attention(q, k, v, mask=attn_mask_inf, is_causal=is_causal)

        attn_output = attn_output.numpy()
        
    return {"result": attn_output}


def main():
    A_TOL = 0.01

    input_data = {
        "q": np.random.rand(2, 3, 4).astype(np.float32),
        "k": np.random.rand(2, 5, 4).astype(np.float32),
        "v": np.random.rand(2, 5, 4).astype(np.float32),
        "key_padding_mask": np.array([[False, False, True, False, True], [False, True, False, False, False]], dtype=np.bool_),
        "need_weights": True,
        "attn_mask": np.random.rand(2, 3, 5).astype(np.float32) * -100,
        "average_attn_weights": True,
        "is_causal": False,
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()