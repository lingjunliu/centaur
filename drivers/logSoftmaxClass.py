import numpy as np

def torch_version(input, cpu=True):
    import torch
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    # input
    x1 = input["dim"]
    x2 = torch.tensor(input["input"])

    # output
    func_obj = torch.nn.LogSoftmax(dim=x1)

    if not cpu:
        x2 = x2.cuda()
        func_obj = func_obj.cuda()
    
    y = func_obj(x2)
    
    if not cpu:
        y = y.cpu()          
    
    return {"logsoftmax": y.numpy()}


def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism()
    if cpu:
        device_string = "/cpu"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # input
        x1 = input["dim"]
        x2 = tf.constant(input["input"])

        # output
        y = tf.nn.log_softmax(x2, axis=x1)   
        
        return {"logsoftmax": y.numpy()}
    
def main():
    # Example input
    input_data = {
        "input": np.random.randn(10, 5).astype(np.float32),  # generate some random data
        "dim": 1  # dimension to perform logcumsumexp along
    }

    # Torch example
    torch_result = torch_version(input_data, cpu=True)
    print("Torch result:", torch_result)

    # TensorFlow example
    tf_result = tensorflow_version(input_data, cpu=True)
    print("TensorFlow result:", tf_result)

    # Assert the results are equal
    if np.allclose(torch_result["logsoftmax"], tf_result["logsoftmax"], atol=1e-2):
        print("equal")
    else:
        print("not equal")

if __name__ == "__main__":
    main()