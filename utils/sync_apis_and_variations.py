import sys
from utils.misc import read_file_in_root, save_file_in_root
from utils.new_api_utils import get_api_suffix

def sync_variation_with_api(lib="torch"):
    variations = read_file_in_root(f"{lib}_variations.txt")
    apis = read_file_in_root(f"{lib}_apis.txt")

    new_variations = []
    for variation in variations:
        api, suffix = get_api_suffix(variation)
        if api in apis:
            new_variations.append(variation)

    save_file_in_root(f"{lib}_variations.txt", "\n".join(new_variations))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        lib = sys.argv[1]
        sync_variation_with_api(lib=lib)
    else:
        sync_variation_with_api(lib="torch")
        sync_variation_with_api(lib="tensorflow")