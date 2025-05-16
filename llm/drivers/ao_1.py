import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    scale = torch.tensor(input_dict.get("scale", 1.0))
    zero_point = torch.tensor(input_dict.get("zero_point", 0))

    if not cpu:
        input_tensor = input_tensor.cuda()
        scale = scale.cuda()
        zero_point = zero_point.cuda()

    q_tensor = torch.quantize_per_tensor(input_tensor, scale=scale.item(), zero_point=zero_point.item(), dtype=torch.quint8)
    
    if not cpu:
        q_tensor = q_tensor.cpu()
    
    return {"result": q_tensor.int_repr().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        scale = input_dict.get("scale", 1.0)
        zero_point = input_dict.get("zero_point", 0)
        # dtype = tf.quint8
        
        # min_range = tf.reduce_min(input_tensor)
        # max_range = tf.reduce_max(input_tensor)
        
        # quantized = tf.quantization.quantize(input_tensor, min_range, max_range, dtype, mode='MIN_COMBINED')
        quantized = tf.round((input_tensor / scale) + zero_point)
        quantized = tf.clip_by_value(quantized, 0, 255)
        result = tf.cast(quantized, dtype=tf.int32).numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "scale": 0.1,
        "zero_point": 10,
        "dtype": np.uint8
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()