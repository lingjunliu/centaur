import numpy as np

def torch_version(input_dict, cpu=True):
    import torch

    precision = input_dict.get("precision", 4)
    threshold = input_dict.get("threshold", 1000.0)
    edgeitems = input_dict.get("edgeitems", 3)
    linewidth = input_dict.get("linewidth", 80)
    profile = input_dict.get("profile", "default")
    sci_mode = input_dict.get("sci_mode", None)
    
    torch.set_printoptions(precision=precision, threshold=threshold, edgeitems=edgeitems, linewidth=linewidth, profile=profile, sci_mode=sci_mode)

    dummy_tensor = torch.randn(2, 3)
    if not cpu:
        dummy_tensor = dummy_tensor.cuda()
    result = dummy_tensor.cpu().numpy()
    return {"result": result}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf

    precision = input_dict.get("precision", 4)
    threshold = input_dict.get("threshold", 1000.0)
    edgeitems = input_dict.get("edgeitems", 3)
    linewidth = input_dict.get("linewidth", 80)
    profile = input_dict.get("profile", "default")
    sci_mode = input_dict.get("sci_mode", None)

    tf.get_logger().setLevel('ERROR')

    if cpu:
        with tf.device('/cpu:0'):
            dummy_tensor = tf.random.normal(shape=(2, 3))
            result = dummy_tensor.numpy()
    else:
        if tf.config.list_physical_devices('GPU'):
            with tf.device('/gpu:0'):
                dummy_tensor = tf.random.normal(shape=(2, 3))
                result = dummy_tensor.numpy()
        else:
            with tf.device('/cpu:0'):
                dummy_tensor = tf.random.normal(shape=(2, 3))
                result = dummy_tensor.numpy()

    return {"result": result}

def main():
    A_TOL = 0.01
    input_data = {
        "precision": 6,
        "threshold": 10000,
        "edgeitems": 5,
        "linewidth": 120,
        "profile": "default",
        "sci_mode": False
    }

    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)


    assert np.allclose(torch_result["result"], tf_result["result"], atol=A_TOL)

    print("Success")

if __name__ == "__main__":
    main()