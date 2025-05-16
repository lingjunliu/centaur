import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"]).unsqueeze(0).unsqueeze(0)
    output_size = input_dict["output_size"]

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    result = torch.nn.functional.adaptive_max_pool1d(input_tensor, output_size)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.squeeze().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        output_size = input_dict["output_size"]

        input_tensor = tf.reshape(input_tensor, [1, 1, -1])
        
        target_len = int(output_size)

        input_shape = tf.shape(input_tensor)
        input_length = tf.cast(input_shape[-1], tf.float32)
        
        
        
        pooled_result = []
        
        if target_len > 0:

            stride = input_length / target_len
            kernel_size = input_length - (target_len - 1) * stride
            stride = int(np.floor(stride.numpy()))
            kernel_size = int(np.ceil(kernel_size.numpy()))


            for i in range(target_len):
                start = i * stride
                end = start + kernel_size
            
                start = int(np.floor(start))
                end = int(np.ceil(end))
                
                if start >= int(input_length):
                    pooled_result.append(np.float32(-np.inf))
                    continue

                if end > int(input_length):
                    end = int(input_length)
                    
                window = input_tensor[0, 0, start:end]
                
                if tf.size(window) == 0:
                  pooled_result.append(np.float32(-np.inf))
                  continue
                
                pooled_result.append(tf.reduce_max(window).numpy())
        else:
            pooled_result = []

        result = np.array(pooled_result, dtype=np.float32)

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "output_size": 2
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32),
        "output_size": 3
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()