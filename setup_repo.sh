HEREDIR=`pwd`
VENV_DIR=${HEREDIR}/venv
echo "Creating venv in: ${VENV_DIR}"

[ -d "${VENV_DIR}" ] && rm -rf "${VENV_DIR}";
virtualenv -p python3.7 ${VENV_DIR}
echo export PYTHONPATH="${HEREDIR}" >> ${VENV_DIR}/bin/activate # set PYTHONPATH to ntt project
source ${VENV_DIR}/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Submodule init/update.."
git submodule init
git submodule update

echo "Downloading data files.."
gdown https://drive.google.com/uc?id=1MaDUdkkl9eSSUI3CB4P3xVN0moNr8yBF
unzip data.zip
rm data.zip
gdown https://drive.google.com/uc?id=1Pq2p1ZY527rGXcKJJJdEgB7lvor1tnDW
unzip _cache.zip
rm _cache.zip

# try it:
#PTOOLS_DIR=${HEREDIR}/ptools
#cd $(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
#echo ${PTOOLS_DIR} > ptools.pth