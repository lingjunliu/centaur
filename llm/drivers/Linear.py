import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input_tensor = torch.tensor(input_dict["input"])
    in_features = input_dict["in_features"]
    out_features = input_dict["out_features"]
    bias = input_dict.get("bias", True)

    linear_layer = torch.nn.Linear(in_features, out_features, bias=bias)

    if "weight" in input_dict:
        linear_layer.weight = torch.nn.Parameter(torch.tensor(input_dict["weight"]))
    if "bias_value" in input_dict and bias:
        linear_layer.bias = torch.nn.Parameter(torch.tensor(input_dict["bias_value"]))
    elif not bias:
      linear_layer.bias = None
    
    if not cpu:
        linear_layer = linear_layer.cuda()
        input_tensor = input_tensor.cuda()

    result = linear_layer(input_tensor)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"].reshape(1, -1))
        in_features = input_dict["in_features"]
        out_features = input_dict["out_features"]
        bias_flag = input_dict.get("bias", True)

        weight_initializer = tf.keras.initializers.Ones()
        bias_initializer = tf.keras.initializers.Zeros()

        if "weight" in input_dict:
            weight_initializer = tf.constant_initializer(input_dict["weight"].transpose())
        if "bias_value" in input_dict and bias_flag:
            bias_initializer = tf.constant_initializer(input_dict["bias_value"])
        elif not bias_flag:
            bias_initializer = None
        
        linear_layer = tf.keras.layers.Dense(
            units=out_features,
            use_bias=bias_flag,
            kernel_initializer=weight_initializer,
            bias_initializer=bias_initializer
        )

        result = linear_layer(input_tensor)
        
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "in_features": 3,
        "out_features": 2,
        "weight": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
        "bias_value": np.array([0.7, 0.8], dtype=np.float32),
        "bias": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data_no_bias = {
        "input": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "in_features": 3,
        "out_features": 2,
        "weight": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
        "bias": False
    }

    torch_result_no_bias = torch_version(input_data_no_bias)
    tf_result_no_bias = tensorflow_version(input_data_no_bias)

    assert np.allclose(torch_result_no_bias["result"], tf_result_no_bias["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()