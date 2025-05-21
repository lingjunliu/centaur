# ------------------------------------------------------------
# COMMAND FORMAT:
#   bash run_harness.sh [-z3 true|false] [-print true|false]
#
# All parameters are optional.
# Defaults: -z3 false, -print false
# ------------------------------------------------------------

PROJECT_DIR=`dirname "$(realpath "$0")"`/..
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# create virtual environment
python -m venv venv
source venv/bin/activate
pip install -r $PROJECT_DIR/requirements.txt
# Tensorflow envrironment variables
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2

z3_flag=false
print_arg=false

while [[ "$1" != "" ]]; do
  case $1 in
    -z3) z3_flag=${2:-false}; shift 2 ;;
    -print) print_arg=${2:-false}; shift 2 ;;
    *) shift ;;
  esac
done

harness="generator.harness$( [ "$z3_flag" == "true" ] && echo "_z3" )"

(cd $PROJECT_DIR;
  PYTHONWARNINGS="ignore" python -m $harness $print_arg
)
