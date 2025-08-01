import numpy as np
import os
import sys
import json

from z3 import *
from .z3 import convert_model_to_abs

def main():    
    api = sys.argv[1]
    lib = sys.argv[2] if len(sys.argv) > 2 else "torch"

    # alias
    if lib == "tensorflow":
        lib = "tf"
    elif lib == "pytorch":
        lib = "torch"

    convert_model_to_abs(api, lib=lib)

if __name__ == "__main__":
    main()
