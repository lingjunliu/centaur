
from utils.new_api_utils import run_api, get_signature
from generator.input_generators import get_abstract_input

generated_inputs = dict()

import copy
import numpy as np
import tempfile
import os
import struct

def tf_experimental_register_filesystem_plugin_inputs():
    """
    Generates inputs for tf.experimental.register_filesystem_plugin.
    This function creates a temporary file with a crafted, minimal ELF header.
    This is necessary to bypass a series of low-level checks performed by the
    underlying dynamic library loader, which would otherwise raise errors like
    'FileNotFoundError', 'file too short', or errors about specific ELF header
    fields. By providing a header that is structurally plausible (correct size,
    magic number, version, and entry sizes), we can ensure the input is valid
    enough to proceed to the library loading stage, where it is expected to fail
    with the documented 'RuntimeError' because it is not a functional plugin.
    """
    list_of_inputs = []

    try:
        # 1. e_ident (16 bytes): ELF identification block.
        e_ident = (
            b'\x7fELF' +    # Magic number
            b'\x02' +       # EI_CLASS: 64-bit
            b'\x01' +       # EI_DATA: Little-endian
            b'\x01' +       # EI_VERSION: EV_CURRENT
            b'\x00' +       # EI_OSABI: System V
            b'\x00' * 8     # Padding
        )

        # 2. Rest of the ELF64_Ehdr (48 bytes), structured to avoid loader errors.
        #    The key fix is setting e_phentsize and e_shentsize to their expected
        #    non-zero values for a 64-bit ELF file.
        elf_header_rest = struct.pack(
            '<HHQIQQIHHHHHH',
            3,      # e_type = ET_DYN (shared object)
            62,     # e_machine = EM_X86_64
            1,      # e_version = EV_CURRENT
            0,      # e_entry = 0 (no entry point)
            0,      # e_phoff = 0 (no program header table)
            0,      # e_shoff = 0 (no section header table)
            0,      # e_flags = 0
            64,     # e_ehsize = 64 (size of this ELF header)
            56,     # e_phentsize = 56 (expected size of a 64-bit program header)
            0,      # e_phnum = 0 (no program headers)
            64,     # e_shentsize = 64 (expected size of a 64-bit section header)
            0,      # e_shnum = 0 (no section headers)
            0       # e_shstrndx = 0
        )

        content = e_ident + elf_header_rest

        with tempfile.NamedTemporaryFile(suffix=".so", delete=False) as temp_file:
            temp_file.write(content)
            plugin_path = temp_file.name

        input_dict = {
            'plugin_location': plugin_path
        }
        list_of_inputs.append(copy.deepcopy(input_dict))

    except Exception:
        # If any error occurs during file creation, return an empty list.
        pass

    return list_of_inputs

generated_inputs["tf.experimental.register_filesystem_plugin"] = tf_experimental_register_filesystem_plugin_inputs()

def check_valid(api, list_of_inputs, lib="tf", suffix=0):
    for idx, input_dict in enumerate(list_of_inputs):
        _ = get_abstract_input(input_dict, get_signature(api, lib=lib, suffix=suffix))
        output = run_api(api, input_dict, cpu=True, lib=lib)
    
    if len(list_of_inputs) == 0:
        raise Exception("No inputs were generated for the API. Please check the input generation code.")

    print("Valid")

if 'tf.experimental.register_filesystem_plugin' not in generated_inputs:
    raise Exception("Output of the input generating function was not assigned to the generated_inputs dictionary to the key 'tf.experimental.register_filesystem_plugin'.")

check_valid('tf.experimental.register_filesystem_plugin', generated_inputs['tf.experimental.register_filesystem_plugin'], lib="tf", suffix=0)
