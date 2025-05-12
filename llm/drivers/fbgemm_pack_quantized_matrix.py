import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"], dtype=torch.float32)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    q_input = torch.quantize_per_tensor(input_tensor, scale=1.0, zero_point=0, dtype=torch.quint8)

    try:
        result = torch.ops.fb.fbgemm_pack_quantized_matrix(q_input)
        result = result.numpy()
    except AttributeError:
        result = q_input.int_repr().numpy()

    if not cpu:
        result = result.cpu()
    
    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        
        min_range = tf.reduce_min(input_tensor)
        max_range = tf.reduce_max(input_tensor)
        
        quantized_tensor, scale, zero_point = tf.quantization.quantize_v2(input_tensor, min_range, max_range, tf.quint8)

        result = tf.cast(quantized_tensor, dtype=tf.float32).numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()