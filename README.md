conda create --name tf python=3.11
conda activate tf
conda install -c apple tensorflow-deps --force-reinstall
python -m pip install tensorflow-macos tensorflow-metal