import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    in1_features = input_dict["in1_features"]
    in2_features = input_dict["in2_features"]
    out_features = input_dict["out_features"]
    input1 = torch.tensor(input_dict["input1"])
    input2 = torch.tensor(input_dict["input2"])
    weight = torch.tensor(input_dict["weight"])
    bias = input_dict.get("bias", None)
    if bias is not None:
        bias = torch.tensor(input_dict["bias"])

    if not cpu:
        input1 = input1.cuda()
        input2 = input2.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    result = torch.nn.functional.bilinear(input1, input2, weight, bias)

    if not cpu:
        result = result.cpu()
    
    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    in1_features = input_dict["in1_features"]
    in2_features = input_dict["in2_features"]
    out_features = input_dict["out_features"]
    input1 = tf.constant(input_dict["input1"])
    input2 = tf.constant(input_dict["input2"])
    weight = tf.constant(input_dict["weight"])
    bias = input_dict.get("bias", None)
    if bias is not None:
        bias = tf.constant(input_dict["bias"])

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        batch_size = tf.shape(input1)[0]

        weight_reshaped = tf.reshape(weight, [out_features, in1_features, in2_features])
        result = tf.zeros([batch_size, out_features], dtype=tf.float32)
        
        for b in range(batch_size):
            x = input1[b]
            y = input2[b]
            
            for i in range(out_features):
                sum_val = tf.reduce_sum(weight_reshaped[i] * tf.matmul(tf.reshape(x, [1, in1_features]), tf.reshape(y, [in2_features, 1])))
                result = tf.tensor_scatter_nd_update(result, [[b, i]], [sum_val])

        if bias is not None:
            result = result + bias

        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "in1_features": 3,
        "in2_features": 4,
        "out_features": 5,
        "input1": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        "input2": np.array([[7, 8, 9, 10], [11, 12, 13, 14]], dtype=np.float32),
        "weight": np.random.rand(5, 3, 4).astype(np.float32),
        "bias": np.random.rand(5).astype(np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()