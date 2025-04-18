THISDIR=`dirname "$(realpath "$0")"`
export PYTHONPATH=$THISDIR:$PYTHONPATH

# create virtual environment (i don't want to mess up abid's code)
source venv/bin/activate

pytest
