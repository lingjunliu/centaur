import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    input_tensor = torch.tensor(input_dict["input"])
    if not cpu:
        input_tensor = input_tensor.cuda()
    
    lazy_linear = torch.nn.LazyLinear(out_features=input_dict.get("out_features", None))
    result = lazy_linear(input_tensor)
    
    if not cpu:
        result = result.cpu()
    
    return {"result": result.detach().numpy(), "weight": lazy_linear.weight.detach().numpy(), "bias": lazy_linear.bias.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"
    
    with tf.device(device_string):
        input_tensor = tf.constant(input_dict["input"])
        
        if "weight" not in input_dict:
            input_dict["weight"] = np.transpose(input_dict["torch_weight"])
            input_dict["bias"] = input_dict["torch_bias"]

        weight = tf.constant(input_dict["weight"])
        bias = tf.constant(input_dict["bias"])
        
        result = tf.matmul(input_tensor, weight) + bias
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.random.rand(1, 10).astype(np.float32),
        "out_features": 5
    }

    torch_result = torch_version(input_data)

    input_data_tf = {
        "input": input_data["input"],
        "out_features": input_data["out_features"],
        "torch_weight": torch_result["weight"],
        "torch_bias": torch_result["bias"]
    }
    
    tf_result = tensorflow_version(input_data_tf)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()