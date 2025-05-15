import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict['input'])
    size = input_dict['size']

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.expand_copy(input_tensor, size)

    if not cpu:
        result = result.cpu()

    return {'result': result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    input_tensor = tf.constant(input_dict['input'])
    size = input_dict['size']

    input_shape = tf.shape(input_tensor)
    num_dims_input = tf.rank(input_tensor)
    num_dims_target = len(size)

    if num_dims_target < num_dims_input:
      raise ValueError("Number of dimensions of target size must be >= input tensor dims")
    
    diff = num_dims_target - num_dims_input

    new_shape = tf.concat([tf.ones([diff], dtype=tf.int32), input_shape], axis=0)
    input_tensor = tf.reshape(input_tensor, new_shape)
    
    multiples = []
    output_shape = []
    
    for i in range(num_dims_target):
        target_size = size[i]
        input_size = tf.shape(input_tensor)[i]

        if target_size == -1:
           raise ValueError("Tensorflow doesn't support '-1' in target sizes")
        
        if target_size < input_size:
            raise ValueError("Target size can't be smaller than input size")
        
        if target_size == input_size:
            multiples.append(1)
        elif input_size == 1:
            multiples.append(target_size)
        else:
            raise ValueError("Target size must be equal or a multiple of input size")
        
        output_shape.append(target_size)

    result = tf.tile(input_tensor, multiples)

    return {'result': result.numpy()}

def main():
    A_TOL = 0.01

    input_data = {
        'input': np.array([1, 2, 3], dtype=np.float32),
        'size': (2, 3)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    input_data = {
        'input': np.array([[1, 2], [3, 4]], dtype=np.float32),
        'size': (2, 2, 2)
    }
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result['result'], tf_result['result'], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()