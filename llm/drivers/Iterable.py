import numpy as np
import torch
import tensorflow as tf

class MockIterable(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def _mha_attention(self, query, key, value, embed_dim_to_check=None):
        
        batch_size = query.shape[0]
        num_heads = query.shape[1]
        head_dim = query.shape[2]
        
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(head_dim)
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        context_vectors = torch.matmul(attention_probs, value)
        
        return context_vectors

def torch_version(input_dict, cpu=True):
    q = torch.tensor(input_dict["q"])
    k = torch.tensor(input_dict["k"])
    v = torch.tensor(input_dict["v"])
    embed_dim_to_check = input_dict.get("embed_dim_to_check", None)

    if not cpu:
        q = q.cuda()
        k = k.cuda()
        v = v.cuda()

    result = MockIterable()._mha_attention(q, k, v, embed_dim_to_check)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        q = tf.constant(input_dict["q"])
        k = tf.constant(input_dict["k"])
        v = tf.constant(input_dict["v"])
        embed_dim_to_check = input_dict.get("embed_dim_to_check", None)

        def _scaled_dot_product_attention(q, k, v, mask=None):
            matmul_qk = tf.matmul(q, k, transpose_b=True)
            dk = tf.cast(tf.shape(k)[-1], tf.float32)
            scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

            if mask is not None:
                scaled_attention_logits += (mask * -1e9)

            attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
            output = tf.matmul(attention_weights, v)

            return output, attention_weights

        result, _ = _scaled_dot_product_attention(q, k, v)

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "q": np.random.rand(2, 8, 64).astype(np.float32),
        "k": np.random.rand(2, 8, 64).astype(np.float32),
        "v": np.random.rand(2, 8, 64).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()