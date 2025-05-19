import numpy as np
import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

def torch_version(input_dict, cpu=True):

    input_tensor = torch.tensor(input_dict["input"])
    lower = input_dict.get("lower", 1/8)
    upper = input_dict.get("upper", 1/3)
    
    if not cpu:
        input_tensor = input_tensor.cuda()
        
    rrelu = torch.nn.RReLU(lower=lower, upper=upper)
    
    training = input_dict.get("training", False)
    
    rrelu.train(training)
    with torch.no_grad():
        if not training:
          result = rrelu(input_tensor)
        else:
          result = rrelu(input_tensor)

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
        input_tensor = tf.constant(input_dict["input"], dtype=tf.float32)
        lower = input_dict.get("lower", 1/8)
        upper = input_dict.get("upper", 1/3)
        training = input_dict.get("training", False)

        if training:
            random_tensor = tf.random.uniform(shape=tf.shape(input_tensor), minval=lower, maxval=upper)
            result = tf.where(input_tensor < 0, input_tensor * random_tensor, input_tensor)
        else:
            alpha = (lower + upper) / 2.0
            result = tf.where(input_tensor < 0, input_tensor * alpha, input_tensor)
        
        result = result.numpy()
    
    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "input": np.array([-0.5, 0.2, -0.1, 0.8, -0.3], dtype=np.float32),
        "lower": 0.1,
        "upper": 0.3,
        "training": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    input_data = {
        "input": np.array([-0.5, 0.2, -0.1, 0.8, -0.3], dtype=np.float32),
        "lower": 0.1,
        "upper": 0.3,
        "training": True
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    
    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()