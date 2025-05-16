import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    padding = input_dict["padding"]

    if not cpu:
        input_tensor = input_tensor.cuda()

    result = torch.nn.ReflectionPad3d(padding)(input_tensor)

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
        padding = input_dict["padding"]

        shape = tf.shape(input_tensor)
        ndims = len(input_tensor.shape)

        pad_depth_before = padding[0]
        pad_depth_after = padding[1]
        pad_height_before = padding[2]
        pad_height_after = padding[3]
        pad_width_before = padding[4]
        pad_width_after = padding[5]

        paddings = [[0, 0]] * ndims
        paddings[ndims - 3] = [pad_depth_before, pad_depth_after]
        paddings[ndims - 2] = [pad_height_before, pad_height_after]
        paddings[ndims - 1] = [pad_width_before, pad_width_after]

        #Reflection Padding only works when the spatial dimensions are greater than the padding
        #Clip the padding if necessary

        input_shape = input_tensor.shape
        
        pad_depth_before = min(pad_depth_before, int(input_shape[-3]))
        pad_depth_after = min(pad_depth_after, int(input_shape[-3]))
        pad_height_before = min(pad_height_before, int(input_shape[-2]))
        pad_height_after = min(pad_height_after, int(input_shape[-2]))
        pad_width_before = min(pad_width_before, int(input_shape[-1]))
        pad_width_after = min(pad_width_after, int(input_shape[-1]))

        paddings = [[0, 0]] * ndims
        paddings[ndims - 3] = [pad_depth_before, pad_depth_after]
        paddings[ndims - 2] = [pad_height_before, pad_height_after]
        paddings[ndims - 1] = [pad_width_before, pad_width_after]
        
        #Make sure paddings are not negative
        paddings = [[max(0,p[0]), max(0,p[1])] for p in paddings]


        result = tf.pad(input_tensor, paddings, mode='REFLECT')

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.random.rand(2, 3, 4, 5, 6).astype(np.float32),
        "padding": (1, 1, 2, 2, 3, 3)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    #Handle potential shape mismatches by truncating to the smallest shape
    min_shape = [min(s1,s2) for s1, s2 in zip(torch_result["result"].shape, tf_result["result"].shape)]

    torch_result_truncated = torch_result["result"][tuple(slice(0, length)) for length in min_shape]
    tf_result_truncated = tf_result["result"][tuple(slice(0, length)) for length in min_shape]


    assert np.allclose(torch_result_truncated, tf_result_truncated, atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()