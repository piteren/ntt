### NTT
NLP Tools Tests on IMDB
 
### setup repo
 * Run **setup_repo.sh** to:
   * create venv
   * init and update submodule
   * download data and cache files
   
`$ ./recreate_venv.sh ~/.venvs/`

try:

cd $(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")

echo /some/library/path > some-library.pth