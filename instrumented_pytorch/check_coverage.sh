llvm_config=${1:-"llvm-config"}

PROJECT_DIR=`dirname "$(realpath "$0")"`/..
if ! command -v python3.12 &> /dev/null; then
    echo "Error: python3.12 is not installed. Please install it before running this script."
    exit 1
fi
python3.12 -m venv venv
source venv/bin/activate
pip install -r $PROJECT_DIR/requirements.txt
# Install instrumented pytorch
if [ ! -f /tmp/foo.txt ]; then  # Download only if not already downloaded
    pip install gdown
    gdown --fuzzy https://drive.google.com/file/d/1GqydzvLO7XTlFXnSum_zhEulJpC2JRwU/view?usp=sharing -O $PROJECT_DIR/instrumented_pytorch/
fi
pip install $PROJECT_DIR/instrumented_pytorch/torch-*

# Test coverage filtering

rm mm.profraw > /dev/null 2>&1
rm mm.profdata > /dev/null 2>&1
rm mm.lcov > /dev/null 2>&1

bindir=$(${llvm_config} --bindir)

if [ -z $bindir ]; then
    echo "Could not run ${llvm_config}. If LLVM is installed in a custom location, please pass the location of the llvm-config binary as an argument like this: bash filter_coverage.sh <location of llvm-config binary>"
    exit 0
else
    echo "LLVM binaries are installed in ${bindir}"
fi

libname=torch
export TORCH_BUILD_DIR=$(pip show "$libname" | grep "Location:" | awk '{print $2}')/${libname}
echo "Using ${libname} from ${TORCH_BUILD_DIR}"

# generate profraw file
LLVM_PROFILE_FILE=mm.profraw python -c "import torch;print(torch.__version__)"
ls -lh mm.profraw

# generate profdata file
${bindir}/llvm-profdata merge -sparse mm.profraw -o mm.profdata
ls -lh mm.profdata

# for linux: the extension is probably .a not .dylib
LIB1=${TORCH_BUILD_DIR}/lib/libtorch_cpu.so
LIB2=${TORCH_BUILD_DIR}/lib/libtorch.so


${bindir}/llvm-cov export -instr-profile=mm.profdata -format=lcov -object $LIB1 $LIB2 > mm.lcov

# check which files were covered
grep "SF:" mm.lcov | sort | uniq | wc -l

# Restore environment
pip install -r $PROJECT_DIR/requirements.txt