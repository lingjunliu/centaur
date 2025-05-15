import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    
    input_tensor = torch.tensor(input_dict['input'])
    groups = input_dict['groups']
    
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    channel_shuffle = torch.nn.ChannelShuffle(groups)
    result = channel_shuffle(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {'result': result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict['input'])
    groups = input_dict['groups']
    
    shape = input_tensor.shape
    N = shape[0]
    C = shape[1]
    
    input_tensor = tf.reshape(input_tensor, [N, groups, C // groups, *shape[2:]])
    input_tensor = tf.transpose(input_tensor, [0, 2, 1, *range(3, len(shape) + 1)])
    input_tensor = tf.reshape(input_tensor, [N, C, *shape[2:]])

    result = input_tensor.numpy()

    return {'result': result}

def main():
    A_TOL = 0.01
    
    input_data = {
        'input': np.arange(1, 17, dtype=np.float32).reshape(1, 4, 2, 2),
        'groups': 2
    }
    
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"
    
    print("Success")

if __name__ == "__main__":
    main()