THISDIR=`dirname "$(realpath "$0")"`
export PYTHONPATH=$THISDIR:$PYTHONPATH

# create virtual environment (i don't want to mess up abid's code)
python -m venv venv
source venv/bin/activate
pip install -r $THISDIR/requirements.txt

pytest
