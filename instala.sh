
conda create --prefix /media/libre/daba

source activate /media/libre/daba

pip install pot

pip install tqdm

pip install plotly

pip install spyder

rm -fr .local/lib/

cd $HOME
git clone https://github.com/gustfrontar/DABA.git

