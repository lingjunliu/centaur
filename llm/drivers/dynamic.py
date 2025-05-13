import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    
    input_tensor = torch.tensor(input_dict["input"])
    qconfig = input_dict.get("qconfig", torch.quantization.default_dynamic_qconfig)

    if not cpu:
        input_tensor = input_tensor.cuda()

    class DynamicQuantWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, x):
            return x
    
    model = DynamicQuantWrapper()
    model.eval()

    if not cpu:
        model = model.cuda()
    
    model.qconfig = qconfig
    torch.quantization.prepare(model, inplace=True)

    model = torch.quantization.convert(model)
    
    result = model(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().dequantize().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = input_dict["input"]
    
    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
        
        s = tf.math.reduce_max(tf.math.abs(input_tensor))
        scale = s / 127.0
        quantized = tf.round(input_tensor / scale)
        clipped = tf.clip_by_value(quantized, -128, 127)
        quantized = tf.cast(clipped, tf.int8)
        
        result = tf.cast(quantized, dtype=tf.float32) * scale
        result = result.numpy()

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