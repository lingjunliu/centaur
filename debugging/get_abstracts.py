import sys, pickle, torch, os
from utils.misc import create_subdir, get_tmp_dir
import logging

logger = logging.getLogger(__name__)

def print_arg(arg, prefix):
    if isinstance(arg, torch.Tensor):        
        try:
            logger.info(f"{prefix} shape: {arg.shape}, dtype: {arg.dtype}, range: [{torch.min(arg)},{torch.max(arg)}]")
        except:
            logger.info(f"{prefix} shape: {arg.shape}, dtype: {arg.dtype}")
    else:
        logger.info(f"{prefix} value: {arg}, dtype: {type(arg)}")

def main():
    api = sys.argv[1]

    output_dir = create_subdir(get_tmp_dir(), "debug_coverage")
    filename_new_br = os.path.join(output_dir, f"{api}.csv")

    if not os.path.exists(filename_new_br):
        print(f"{filename_new_br} does not exist")
        return
    
    logfile = f'{output_dir}/{api}_abstracts.log'
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,                                     # Minimum log level
        format='%(message)s',                                   # Log format
        filename=logfile,                                       # Log file path
        filemode="w"                                            # Append/Write mode
    )

    with open(filename_new_br, "r") as f:
        list_of_files = [line.strip().split(',') for line in f.readlines()]
    
    for file, new_branches in list_of_files:
        with open(file, "rb") as f:
            pkl_dict = pickle.load(f)
            for torch_api, input_dict in pkl_dict.items():
                logger.info(f"New Branches: {new_branches} | API: {torch_api}")
                logger.info("args:")
                count = 0
                for arg in input_dict['args']:
                    count += 1
                    print_arg(arg, f"{count} |")
                
                if len(input_dict['kwargs']) > 0:
                    logger.info("kwargs:")
                for name, kwarg in input_dict['kwargs'].items():
                    count += 1
                    print_arg(kwarg, f"{count} | name: {name},")
                
                logger.info("")

    print(f"Logged abstract info to {logfile}")
if __name__ == "__main__":
    main()