import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    in1_features = input_dict["in1_features"]
    in2_features = input_dict["in2_features"]
    out_features = input_dict["out_features"]
    bias = input_dict.get("bias", True)
    
    input1 = torch.tensor(input_dict["input1"])
    input2 = torch.tensor(input_dict["input2"])
    
    m = torch.nn.Bilinear(in1_features, in2_features, out_features, bias=bias)

    if not cpu:
        input1 = input1.cuda()
        input2 = input2.cuda()
        m = m.cuda()
    
    output = m(input1, input2)
    
    if not cpu:
        output = output.cpu()
    
    return {"result": output.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    in1_features = input_dict["in1_features"]
    in2_features = input_dict["in2_features"]
    out_features = input_dict["out_features"]
    bias = input_dict.get("bias", True)

    input1 = tf.constant(input_dict["input1"])
    input2 = tf.constant(input_dict["input2"])

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        w_shape = (out_features, in1_features, in2_features)
        k = 1 / in1_features
        limit = np.sqrt(k)
        weight_values = np.random.uniform(-limit, limit, size=w_shape).astype(np.float32)
        weight = tf.Variable(weight_values)
        
        output = tf.einsum('ai,ijk,ak->aj', input1, weight, input2)

        if bias:
            b_shape = (out_features,)
            bias_values = np.random.uniform(-limit, limit, size=b_shape).astype(np.float32)
            bias_var = tf.Variable(bias_values)
            output = tf.add(output, bias_var)
        
        output = output.numpy()
    
    return {"result": output}

def main():
    A_TOL = 0.01
    input_data = {
        "in1_features": 20,
        "in2_features": 30,
        "out_features": 40,
        "input1": np.random.randn(128, 20).astype(np.float32),
        "input2": np.random.randn(128, 30).astype(np.float32),
        "bias": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "in1_features": 5,
        "in2_features": 7,
        "out_features": 10,
        "input1": np.random.randn(64, 5).astype(np.float32),
        "input2": np.random.randn(64, 7).astype(np.float32),
        "bias": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"
    print("Success")

if __name__ == "__main__":
    main()