import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    from torch.nn.quantized.modules import Quantize

    input_tensor = torch.tensor(input_dict["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    scale = 0.1
    zero_point = 0
    dtype = torch.quint8
    
    mod = Quantize(scale=scale, zero_point=zero_point, dtype=dtype)
    result = mod(input_tensor)

    if not cpu:
        result = result.cpu()
    
    return {"result": result.dequantize().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)

        scale = 0.1
        zero_point = 0

        q_params = tf.quantization.quantize(input_tensor, clip_value_min=-1.0, clip_value_max=1.0, round_mode="HALF_UP", narrow_range=False, axis=None, signed_input=True)
        
        quantized_tensor = q_params[0]

        dequantized_tensor = tf.cast(quantized_tensor, tf.float32)

        result = dequantized_tensor.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.1
    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()