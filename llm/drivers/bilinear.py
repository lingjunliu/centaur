import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    input1 = torch.tensor(input_dict["input1"])
    input2 = torch.tensor(input_dict["input2"])
    weight = torch.tensor(input_dict["weight"])
    bias = input_dict.get("bias", None)
    if bias is not None:
        bias = torch.tensor(bias)

    if not cpu:
        input1 = input1.cuda()
        input2 = input2.cuda()
        weight = weight.cuda()
        if bias is not None:
            bias = bias.cuda()

    result = torch.bilinear(input1, input2, weight, bias=bias)

    if not cpu:
        result = result.cpu()

    return {"result": result.numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input1 = tf.constant(input_dict["input1"])
        input2 = tf.constant(input_dict["input2"])
        weight = tf.constant(input_dict["weight"])
        bias_np = input_dict.get("bias", None)
        bias = tf.constant(bias_np) if bias_np is not None else None

        input1_shape = input1.shape
        input2_shape = input2.shape
        weight_shape = weight.shape
        
        input1_reshaped = tf.reshape(input1, (-1, input1_shape[-1]))
        input2_reshaped = tf.reshape(input2, (-1, input2_shape[-1]))

        # Ensure correct shape for weight matrix multiplication
        w = weight.numpy()
        w = w.reshape(input1.shape[-1], input2.shape[-1], -1)
        weight = tf.constant(w)

        output = tf.einsum('ai,bij,bj->ab', input1_reshaped, weight, input2_reshaped)
        output = tf.reshape(output, (-1, weight_shape[-1]))
        output = tf.reduce_sum(output, axis=1)

        if bias is not None:
            output = output + bias

        output = tf.reshape(output, input1_shape[:-1])

        result = output.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "input1": np.random.rand(2, 3, 4).astype(np.float32),
        "input2": np.random.rand(2, 3, 5).astype(np.float32),
        "weight": np.random.rand(4, 5, 7).astype(np.float32),
        "bias": np.random.rand(7).astype(np.float32)
    }
    
    # Correcting the shape of weight and bias to satisfy torch.bilinear's requirements
    input_data["input1"] = np.random.rand(2, 3, 4).astype(np.float32)
    input_data["input2"] = np.random.rand(2, 3, 5).astype(np.float32)
    input_data["weight"] = np.random.rand(4, 5, 7).astype(np.float32)
    input_data["bias"] = np.random.rand(7).astype(np.float32)
    
    
    input_size1 = input_data["input1"].shape[-1]
    input_size2 = input_data["input2"].shape[-1]
    output_size = 7 #input_data["weight"].shape[-1]


    # Correcting the shape of weight to satisfy torch.bilinear's requirements
    input_data["weight"] = np.random.rand(input_size1, input_size2, output_size).astype(np.float32) # Example: output size is 10
    input_data["bias"] = np.random.rand(output_size).astype(np.float32) 
    

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()