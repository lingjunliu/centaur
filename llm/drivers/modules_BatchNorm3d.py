import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict['input'])
    num_features = input_dict['num_features']
    eps = input_dict.get('eps', 1e-05)
    momentum = input_dict.get('momentum', 0.1)
    affine = input_dict.get('affine', True)
    track_running_stats = input_dict.get('track_running_stats', True)

    if not cpu:
        input_tensor = input_tensor.cuda()

    bn = torch.nn.BatchNorm3d(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

    if 'running_mean' in input_dict:
        bn.running_mean = torch.tensor(input_dict['running_mean'])
        if not cpu:
            bn.running_mean = bn.running_mean.cuda()

    if 'running_var' in input_dict:
        bn.running_var = torch.tensor(input_dict['running_var'])
        if not cpu:
            bn.running_var = bn.running_var.cuda()
        
    if affine:
        if 'weight' in input_dict:
            bn.weight = torch.nn.Parameter(torch.tensor(input_dict['weight']))
            if not cpu:
                bn.weight = torch.nn.Parameter(bn.weight.cuda())
        if 'bias' in input_dict:
            bn.bias = torch.nn.Parameter(torch.tensor(input_dict['bias']))
            if not cpu:
                bn.bias = torch.nn.Parameter(bn.bias.cuda())

    bn.eval()
    with torch.no_grad():
        result = bn(input_tensor)

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
        eps = input_dict.get('eps', 1e-05)
        momentum = input_dict.get('momentum', 0.1)
        affine = input_dict.get('affine', True)
        track_running_stats = input_dict.get('track_running_stats', True)

        gamma = None
        beta = None
        moving_mean = None
        moving_variance = None

        if affine:
            if 'weight' in input_dict:
                gamma = tf.constant(input_dict['weight'])
            else:
                gamma = tf.ones([num_features], dtype=tf.float32)
            if 'bias' in input_dict:
                beta = tf.constant(input_dict['bias'])
            else:
                beta = tf.zeros([num_features], dtype=tf.float32)

        if track_running_stats:
            if 'running_mean' in input_dict:
                moving_mean = tf.constant(input_dict['running_mean'])
            else:
                moving_mean = tf.zeros([num_features], dtype=tf.float32)
            if 'running_var' in input_dict:
                moving_variance = tf.constant(input_dict['running_var'])
            else:
                moving_variance = tf.ones([num_features], dtype=tf.float32)
        else:
            moving_mean = None
            moving_variance = None

        bn = tf.keras.layers.BatchNormalization(
            axis=1,
            epsilon=eps,
            momentum=momentum,
            beta_initializer=tf.keras.initializers.Constant(beta) if beta is not None else 'zeros',
            gamma_initializer=tf.keras.initializers.Constant(gamma) if gamma is not None else 'ones',
            moving_mean_initializer=tf.keras.initializers.Constant(moving_mean) if moving_mean is not None else 'zeros',
            moving_variance_initializer=tf.keras.initializers.Constant(moving_variance) if moving_variance is not None else 'ones',
            trainable=False
        )

        result = bn(input_tensor, training=False)
        
        result = result.numpy()

    return {'result': result}

def main():
    A_TOL = 0.01

    input_data = {
        'input': np.random.rand(2, 3, 4, 5, 6).astype(np.float32),
        'num_features': 3,
        'eps': 1e-05,
        'momentum': 0.1,
        'affine': True,
        'track_running_stats': True,
        'running_mean': np.random.rand(3).astype(np.float32),
        'running_var': np.random.rand(3).astype(np.float32),
        'weight': np.random.rand(3).astype(np.float32),
        'bias': np.random.rand(3).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()