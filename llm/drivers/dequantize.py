import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["tensor"], dtype=torch.float32)

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    input_tensor = torch.quantize_per_tensor(input_tensor, scale=1.0, zero_point=0, dtype=torch.qint8)

    result = torch.dequantize(input_tensor)
    
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
        input_tensor = tf.constant(input_dict["tensor"], dtype=tf.float32)
        
        result = tf.cast(input_tensor, dtype=tf.float32)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "tensor": np.array([1, 2, 3, 4], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "tensor": [np.array([1, 2, 3, 4], dtype=np.float32), np.array([5,6,7,8], dtype=np.float32)]
    }
    
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    
    def torch_version_list(input_dict, cpu=True):
        import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    
        input_tensors = [torch.tensor(tensor, dtype=torch.float32) for tensor in input_dict["tensor"]]
        
        if not cpu:
            input_tensors = [tensor.cuda() for tensor in input_tensors]
        
        quantized_tensors = []
        for tensor in input_tensors:
            quantized_tensors.append(torch.quantize_per_tensor(tensor, scale=1.0, zero_point=0, dtype=torch.qint8))

        result = torch.dequantize(quantized_tensors)
        
        if not cpu:
            result = [r.cpu() for r in result]
        
        return {"result": [r.numpy() for r in result]}
    
    def tensorflow_version_list(input_dict, cpu=True):
        import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    
        if cpu:
            device_string = "/cpu:0"
        else:
            device_string = "/gpu:0"
    
        with tf.device(device_string):
            input_tensors = [tf.constant(tensor, dtype=tf.float32) for tensor in input_dict["tensor"]]
            
            result = [tf.cast(tensor, dtype=tf.float32).numpy() for tensor in input_tensors]
            
        return {"result": result}

    torch_result = torch_version_list(input_data)
    tf_result = tensorflow_version_list(input_data)

    for i in range(len(torch_result["result"])):
        assert np.allclose(torch_result["result"][i], tf_result["result"][i], atol=A_TOL), "Results do not match"
        
    print("Success")

if __name__ == "__main__":
    main()