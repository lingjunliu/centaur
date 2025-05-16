import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict["bias"]) if "bias" in input_dict else None
    pad = input_dict.get("pad", 0)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()
    
    result = torch.conv_tbc(input_tensor, weight, bias, pad)
    
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
        weight = tf.constant(input_dict["weight"])
        bias = tf.constant(input_dict["bias"]) if "bias" in input_dict else None
        pad = input_dict.get("pad", 0)

        input_shape = input_tensor.shape
        weight_shape = weight.shape

        padded_input = tf.pad(input_tensor, [[pad, pad], [0, 0], [0, 0]])

        input_len = input_shape[0] + 2 * pad
        weight_len = weight_shape[0]

        output_len = input_len - weight_len + 1

        output = []
        for i in range(output_len):
            slice_start = i
            slice_end = i + weight_len
            
            input_slice = padded_input[slice_start:slice_end]
            
            conv_result = tf.reduce_sum(input_slice * weight, axis=[0])
            if bias is not None:
                conv_result = conv_result + bias
            output.append(conv_result)
        
        output = tf.stack(output)
        result = output.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=np.float32).transpose((1, 0, 2)),
        "weight": np.array([[[0.1, 0.2]], [[0.3, 0.4]]], dtype=np.float32).transpose((1, 0, 2)),
        "bias": np.array([0.5, 0.6], dtype=np.float32),
        "pad": 0
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    tf_result['result'] = tf_result['result'].transpose((1, 0))

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=np.float32).transpose((1, 0, 2)),
        "weight": np.array([[[0.1, 0.2]], [[0.3, 0.4]]], dtype=np.float32).transpose((1, 0, 2)),
        "bias": np.array([0.5, 0.6], dtype=np.float32),
        "pad": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    tf_result['result'] = tf_result['result'].transpose((1, 0))

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"


    print("Success")

if __name__ == "__main__":
    main()