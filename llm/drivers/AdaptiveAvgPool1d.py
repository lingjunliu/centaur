import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    output_size = input_dict["output_size"]
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    adaptive_avg_pool1d = torch.nn.AdaptiveAvgPool1d(output_size)
    result = adaptive_avg_pool1d(input_tensor)
    
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
        output_size = input_dict["output_size"]

        input_shape = tf.shape(input_tensor)
        num_channels = input_shape[0]
        input_length = input_shape[1]
        
        target_length = output_size

        def compute_start_end(index):
            start = index * input_length // target_length
            end = (index + 1) * input_length // target_length
            return start, end

        pooled_values = []
        for i in range(target_length):
            start, end = compute_start_end(i)
            window = input_tensor[:, start:end]
            avg_val = tf.reduce_mean(window, axis=1, keepdims=True)
            pooled_values.append(avg_val)

        result = tf.concat(pooled_values, axis=1)

        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float32),
        "output_size": 3
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()