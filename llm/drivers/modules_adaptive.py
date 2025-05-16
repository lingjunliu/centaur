import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    output_size = input_dict["output_size"]

    if not cpu:
        input_tensor = input_tensor.cuda()
    
    adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(output_size)
    result = adaptive_avg_pool2d(input_tensor)
    
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

        ksize = [input_tensor.shape[1] // output_size[0], input_tensor.shape[2] // output_size[1]]
        strides = [input_tensor.shape[1] // output_size[0], input_tensor.shape[2] // output_size[1]]
        
        result = tf.nn.avg_pool(tf.expand_dims(input_tensor, 0), ksize=[1, ksize[0], ksize[1], 1], strides=[1, strides[0], strides[1], 1], padding='VALID')
        result = result.numpy()
        result = np.squeeze(result, axis=0)

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(8, 8, 3).astype(np.float32),
        "output_size": (4, 4)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()