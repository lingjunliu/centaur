import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    
    input_tensor = torch.tensor(input_dict['input'], requires_grad=True)
    bn = torch.nn.BatchNorm3d(input_dict['num_features'])

    if not cpu:
        input_tensor = input_tensor.cuda()
        bn = bn.cuda()

    result = torch.nn.functional.batch_norm(
        input_tensor, 
        running_mean=bn.running_mean,
        running_var=bn.running_var, 
        weight=bn.weight,
        bias=bn.bias,
        training=input_dict.get("training", False),
        momentum=input_dict.get("momentum", 0.1),
        eps=input_dict.get("eps", 1e-05)
    )

    if not cpu:
        result = result.cpu()
    
    return {'result': result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict['input'])
        num_features = input_dict['num_features']
        training = input_dict.get("training", False)
        momentum = input_dict.get("momentum", 0.1)
        epsilon = input_dict.get("eps", 1e-05)

        gamma = tf.Variable(tf.ones([num_features]), trainable=False)
        beta = tf.Variable(tf.zeros([num_features]), trainable=False)
        moving_mean = tf.Variable(tf.zeros([num_features]), trainable=False)
        moving_variance = tf.Variable(tf.ones([num_features]), trainable=False)
        
        axes = list(range(len(input_tensor.shape)))
        axes = axes[0:len(axes)-1]
        
        if training:
            batch_mean, batch_variance = tf.nn.moments(input_tensor, axes=axes, keepdims=False)
            
            def update_moving_average():
                update_moving_mean = tf.compat.v1.assign(moving_mean, moving_mean * momentum + batch_mean * (1 - momentum))
                update_moving_variance = tf.compat.v1.assign(moving_variance, moving_variance * momentum + batch_variance * (1 - momentum))
                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    return tf.identity(batch_mean), tf.identity(batch_variance)
            
            batch_mean, batch_variance = update_moving_average()
        else:
            batch_mean, batch_variance = moving_mean, moving_variance
        
        result = tf.nn.batch_normalization(input_tensor, batch_mean, batch_variance, beta, gamma, epsilon)
        
        result = result.numpy()

    return {'result': result}

def main():
    A_TOL = 0.01

    input_data = {
        'input': np.random.rand(2, 3, 4, 5, 6).astype(np.float32),
        'num_features': 3,
        'training': True,
        'momentum': 0.1,
        'eps': 1e-05
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()