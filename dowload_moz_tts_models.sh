#!/usr/bin/env bash
source ./jarvis_virtualenv/bin/activate

TTS_MODEL_URL="https://www.dropbox.com/sh/z62xzmh6qvmn6r0/AAD2e_aoL6nlwj24gLvsUtj6a?dl=0"
PWGAN_MODEL_URL="https://www.dropbox.com/sh/fz8iixkhv68zsb4/AABlrNomybrGIinOrgLhZeosa?dl=0"
TTS_REPO="https://github.com/nvladimirovi/TTS.git"

echo $TTS_MODEL_URL

wget -O tts_model.zip --no-check-certificate $TTS_MODEL_URL
wget -O pwgan_model.zip --no-check-certificate $PWGAN_MODEL_URL

sudo apt-get install espeak

git clone $TTS_REPO
cd ./TTS
git checkout -b feature/nvi/build-my-1st-great-model
pip3 install -r requirements.txt
python setup.py install

cd ../
git clone https://github.com/erogol/ParallelWaveGAN
cd ./ParallelWaveGAN
git checkout fca88f9
pip install .
cd ..
