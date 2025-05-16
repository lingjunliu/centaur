import numpy as np

def torch_version(input_dict, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    in1_features = torch.tensor(input_dict["in1_features"])
    in2_features = torch.tensor(input_dict["in2_features"])
    out_features = torch.tensor(input_dict["out_features"])
    input1 = torch.tensor(input_dict["input1"], requires_grad=False)
    input2 = torch.tensor(input_dict["input2"], requires_grad=False)

    if not cpu:
        in1_features = in1_features.cuda()
        in2_features = in2_features.cuda()
        out_features = out_features.cuda()
        input1 = input1.cuda()
        input2 = input2.cuda()

    bilinear = torch.nn.Bilinear(in1_features.item(), in2_features.item(), out_features.item())
    
    with torch.no_grad():
        result = bilinear(input1, input2)

    if not cpu:
        result = result.cpu()

    return {"result": result.detach().numpy()}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()

    in1_features = input_dict["in1_features"]
    in2_features = input_dict["in2_features"]
    out_features = input_dict["out_features"]
    input1 = input_dict["input1"]
    input2 = input_dict["input2"]

    if cpu:
        device_string = "/cpu:0"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        input1_tf = tf.constant(input1, dtype=tf.float32)
        input2_tf = tf.constant(input2, dtype=tf.float32)

        W = tf.Variable(tf.random.normal((in1_features, in2_features, out_features), dtype=tf.float32))
        b = tf.Variable(tf.zeros((out_features,), dtype=tf.float32))

        input1_expanded = tf.expand_dims(input1_tf, 0)
        input2_expanded = tf.expand_dims(input2_tf, 0)
        
        batch_matmul_result = tf.einsum('ai,ijk,ak->aj', input1_expanded, W, input2_expanded)
        result = batch_matmul_result[0] + b
        result = result.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01

    input_data = {
        "in1_features": np.array(2, dtype=np.int32),
        "in2_features": np.array(3, dtype=np.int32),
        "out_features": np.array(4, dtype=np.int32),
        "input1": np.array([1.0, 2.0], dtype=np.float32),
        "input2": np.array([3.0, 4.0, 5.0], dtype=np.float32)
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)

    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL), "Results do not match"

    print("Success")

if __name__ == "__main__":
    main()