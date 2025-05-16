import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    scale = input_dict.get("scale", 1.0)
    zero_point = input_dict.get("zero_point", 0)
    quant_min = input_dict.get("quant_min", 0)
    quant_max = input_dict.get("quant_max", 255)

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.quantize_per_tensor(input_tensor, scale, zero_point, torch.quint8).int_repr()

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
        scale = input_dict.get("scale", 1.0)
        zero_point = input_dict.get("zero_point", 0)
        quant_min = input_dict.get("quant_min", 0)
        quant_max = input_dict.get("quant_max", 255)
        
        quantized = tf.clip_by_value(tf.round(input_tensor / scale + zero_point), quant_min, quant_max)
        result = quantized.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([0.0202, 1.0985, 1.3506, -0.6056], dtype=np.float32),
        "scale": 0.1,
        "zero_point": 10,
        "quant_min": 0,
        "quant_max": 255
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()