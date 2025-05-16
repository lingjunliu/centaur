import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    attn_mask = torch.tensor(input_dict["attn_mask"])

    if not cpu:
        input_tensor = input_tensor.cuda()
        attn_mask = attn_mask.cuda()

    result = torch.nn.functional.scaled_dot_product_attention(input_tensor, input_tensor, input_tensor, attn_mask=attn_mask)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        attn_mask = tf.constant(input_dict["attn_mask"], dtype=tf.float32)

        q = input_tensor
        k = input_tensor
        v = input_tensor

        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        masked_attention_logits = scaled_attention_logits + attn_mask
        
        attention_weights = tf.nn.softmax(masked_attention_logits, axis=-1)

        output = tf.matmul(attention_weights, v)

        result = output.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 4, 32).astype(np.float32),
        "attn_mask": np.random.randint(0, 2, size=(1, 4, 4)).astype(np.float32)
    }
    input_data["attn_mask"] = (input_data["attn_mask"] * -1e9) * (input_data["attn_mask"] == 0)
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()