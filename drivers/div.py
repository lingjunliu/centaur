
def torch_version(input, cpu=True):
    import torch
    # input
    x1 = torch.tensor(input["input"])
    x2 = torch.tensor(input["other"])

    if not cpu:
        x1 = x1.cuda()
        x2 = x2.cuda()

    # output
    y = torch.div(x1, x2)

    if not cpu:
        y = y.cpu()           
    
    return {"div": y.numpy()}


def tensorflow_version(input, cpu=True):
    import tensorflow as tf
    if cpu:
        device_string = "/cpu"
    else:
        device_string = "/gpu:0"

    with tf.device(device_string):
        # input
        x1 = tf.constant(input["input"])
        x2 = tf.constant(input["other"])

        # output        
        y = tf.divide(x1, x2)
        
        return {"div": y.numpy()}
