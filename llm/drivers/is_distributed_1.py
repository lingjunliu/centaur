import numpy as np
import torch
import os

def torch_version(input_dict, cpu=True):
    if 'backend' in input_dict:
        backend = input_dict['backend']
    else:
        backend = 'gloo' if torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else 'gloo'

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    try:
        torch.distributed.init_process_group(
            backend=backend,
            rank=0,
            world_size=1,
            init_method="env://"
        )
        result = torch.distributed.is_initialized()
    except RuntimeError:
        result = False
    finally:
        if torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except RuntimeError:
                pass
        if 'MASTER_ADDR' in os.environ:
            del os.environ['MASTER_ADDR']
        if 'MASTER_PORT' in os.environ:
            del os.environ['MASTER_PORT']
    return {"result": np.array(result)}

def tensorflow_version(input_dict, cpu=True):
    import tensorflow as tf
    try:
        import horovod.tensorflow as hvd
        hvd.init()
        initialized = True
        hvd.shutdown()

    except ImportError:
        initialized = False
    except RuntimeError:
        initialized = False
    return {"result": np.array(initialized)}

def main():
    input_data = {}
    torch_result = torch_version(input_data)
    tf_result = tensorflow_version(input_data)
    assert np.allclose(torch_result["result"], tf_result["result"])
    print("Success")

if __name__ == "__main__":
    main()