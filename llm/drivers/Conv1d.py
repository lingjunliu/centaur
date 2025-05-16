import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    input_tensor = torch.tensor(input_dict["input"])
    weight = torch.tensor(input_dict["weight"])
    bias = torch.tensor(input_dict.get("bias", np.zeros(input_dict["weight"].shape[0]))) if "bias" in input_dict else None
    stride = input_dict.get("stride", 1)
    padding = input_dict.get("padding", 0)
    dilation = input_dict.get("dilation", 1)
    groups = input_dict.get("groups", 1)

    if not cpu:
        input_tensor = input_tensor.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()
    
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    
    result = torch.nn.functional.conv1d(input_tensor, weight, bias, stride, padding, dilation, groups).squeeze(0).squeeze(0)
    
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
        bias = tf.constant(input_dict.get("bias", np.zeros(input_dict["weight"].shape[0]))) if "bias" in input_dict else None
        stride = input_dict.get("stride", 1)
        padding = input_dict.get("padding", 0)
        dilation = input_dict.get("dilation", 1)
        groups = input_dict.get("groups", 1)
        
        input_tensor_expanded = tf.expand_dims(input_tensor, axis=0)
        input_tensor_expanded = tf.expand_dims(input_tensor_expanded, axis=2)
        weight_expanded = tf.transpose(weight, perm=[2, 1, 0])

        result = tf.nn.conv1d(
            input_tensor_expanded,
            filters=weight_expanded,
            stride=stride,
            padding="VALID",
        )

        if padding > 0:
            pad_before = padding * dilation
            pad_after = padding * dilation
            padding_config = [[0, 0], [pad_before, pad_after], [0, 0]]
            result = tf.pad(result, padding_config)

        if dilation > 1:
            original_length = tf.shape(result)[1]
            new_length = (original_length - 1) * dilation + 1
            
            indices = tf.range(0, original_length) * dilation
            
            updates = result
            shape = [1, new_length, 1]
            
            sparse_tensor = tf.SparseTensor(
                indices=tf.expand_dims(indices, axis=1),
                values=tf.reshape(updates, [-1]),
                dense_shape=[new_length]
            )
            dense_tensor = tf.sparse.to_dense(sparse_tensor, default_value=0)
            result = tf.reshape(dense_tensor, [1, new_length, 1])

        if bias is not None:
            result = tf.nn.bias_add(result, bias)
        
        result = tf.squeeze(result, axis=0).numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "weight": np.array([[[0.1], [0.2], [0.3]]], dtype=np.float32),
        "bias": np.array([0.5], dtype=np.float32),
        "stride": 1,
        "padding": 0,
        "dilation": 1,
        "groups": 1
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()