echo "Cleaning any existing environment"
conda deactivate 
conda remove -y -n image_seg_study --all

conda create -y -n image_seg_study python=3.10.13 -c conda-forge
conda activate image_seg_study

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning==2.3.1
pip install ipython
pip install matplotlib
pip install fire