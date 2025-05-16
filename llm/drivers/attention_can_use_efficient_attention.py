import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    
    input_tensor = torch.tensor(input_dict["input_tensor"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.nn.functional.scaled_dot_product_attention(input_tensor, input_tensor, input_tensor)
    
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
        input_tensor = tf.constant(input_dict["input_tensor"])
        
        q = input_tensor
        k = input_tensor
        v = input_tensor
        
        d_k = tf.cast(tf.shape(k)[-1], tf.float32)
        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(d_k)
        attention_weights = tf.nn.softmax(scores, axis=-1)
        result = tf.matmul(attention_weights, v)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input_tensor": np.random.rand(1, 4, 64).astype(np.float32),
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()