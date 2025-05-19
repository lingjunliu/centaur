import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    
    input_tensor = torch.tensor(input_dict["input"])
    
    if not cpu:
        input_tensor = input_tensor.cuda()

    lazy_batchnorm = torch.nn.LazyBatchNorm2d()

    if not cpu:
        lazy_batchnorm = lazy_batchnorm.cuda()
        
    result = lazy_batchnorm(input_tensor)

    if not cpu:
        result = result.cpu()

    return {'result': result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.convert_to_tensor(input_dict["input"], dtype=tf.float32)
        
        if len(input_tensor.shape) == 2:
            input_tensor = tf.expand_dims(input_tensor, axis=0)
            input_tensor = tf.expand_dims(input_tensor, axis=0)

        if len(input_tensor.shape) == 3:
            input_tensor = tf.expand_dims(input_tensor, axis=0)
        
        
        input_shape = input_tensor.shape
        
        if len(input_shape) == 4:
            axes = [0,2,3]
        else:
            axes = [0]
        
        mean, variance = tf.nn.moments(input_tensor, axes=axes, keepdims=True)

        epsilon = 1e-5
        
        result = tf.nn.batch_normalization(
            input_tensor,
            mean,
            variance,
            offset=tf.zeros(mean.shape[-1], dtype=tf.float32),
            scale=tf.ones(mean.shape[-1], dtype=tf.float32),
            variance_epsilon=epsilon
        )
        
        result = result.numpy()
    return {'result': result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(1, 3, 32, 32).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()