import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict['input'])
    running_mean = torch.tensor(input_dict['running_mean'])
    running_var = torch.tensor(input_dict['running_var'])
    weight = torch.tensor(input_dict['weight'])
    bias = torch.tensor(input_dict['bias'])
    training = input_dict.get('training', False)
    momentum = input_dict.get('momentum', 0.1)
    eps = input_dict.get('eps', 1e-05)

    if not cpu:
        input_tensor = input_tensor.cuda()
        running_mean = running_mean.cuda()
        running_var = running_var.cuda()
        weight = weight.cuda()
        bias = bias.cuda()

    result = torch.batch_norm(
        input_tensor,
        running_mean,
        running_var,
        weight,
        bias,
        training,
        momentum,
        eps,
        cudnn_enabled=True
    )

    if not cpu:
        result = result.cpu()

    return {'result': result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict['input'], dtype=tf.float32)
        running_mean = tf.constant(input_dict['running_mean'], dtype=tf.float32)
        running_var = tf.constant(input_dict['running_var'], dtype=tf.float32)
        weight = tf.constant(input_dict['weight'], dtype=tf.float32)
        bias = tf.constant(input_dict['bias'], dtype=tf.float32)
        training = input_dict.get('training', False)
        momentum = input_dict.get('momentum', 0.1)
        eps = input_dict.get('eps', 1e-05)

        if training:
            mean, variance = tf.nn.moments(input_tensor, axes=[0], keepdims=False)
            
            running_mean = tf.Variable(running_mean)
            running_var = tf.Variable(running_var)

            normalized = (input_tensor - mean) / tf.sqrt(variance + eps)
            result = weight * normalized + bias

            running_mean.assign(running_mean * (1 - momentum) + mean * momentum)
            running_var.assign(running_var * (1 - momentum) + variance * momentum)

        else:
            normalized = (input_tensor - running_mean) / tf.sqrt(running_var + eps)
            result = weight * normalized + bias

        result = result.numpy()

    return {'result': result}

def main():
    A_TOL = 0.01

    input_data = {
        'input': np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32),
        'running_mean': np.array([0.0], dtype=np.float32),
        'running_var': np.array([1.0], dtype=np.float32),
        'weight': np.array([1.0], dtype=np.float32),
        'bias': np.array([0.0], dtype=np.float32),
        'training': False,
        'momentum': 0.1,
        'eps': 1e-05
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), 'Results do not match'

    input_data['training'] = True
    input_data['running_mean'] = np.array([0.0], dtype=np.float32)
    input_data['running_var'] = np.array([1.0], dtype=np.float32)

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), 'Results do not match'


    print('Success')

if __name__ == '__main__':
    main()