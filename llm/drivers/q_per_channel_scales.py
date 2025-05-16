import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    scales = torch.tensor(input_dict["scales"])
    dim = input_dict.get("dim", 0)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        scales = scales.cuda()

    zero_points = torch.zeros(scales.shape, dtype=torch.int32)
        
    q_tensor = torch.quantize_per_channel(input_tensor, scales=scales, zero_points=zero_points, axis=dim, quant_min=-128, quant_max=127, dtype=torch.qint8)

    result = q_tensor.dequantize()
    
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
        input_tensor = tf.constant(input_dict["input"])
        scales = tf.constant(input_dict["scales"])
        dim = input_dict.get("dim", 0)
        
        input_shape = tf.shape(input_tensor)
        scales_shape = tf.shape(scales)

        rank = len(input_tensor.shape)
        dim = dim % rank
        
        target_shape = [1] * rank
        target_shape[dim] = input_shape[dim]
        scales_reshaped = tf.reshape(scales, target_shape)
        
        result = tf.multiply(input_tensor, scales_reshaped)
        
        result = result.numpy()
        
    return {"result": result}

def main():
    A_TOL = 0.1
    
    input_data = {
        "input": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "scales": np.array([0.5, 0.25], dtype=np.float32),
        "dim": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()