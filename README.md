# ASPL_REM
##### Table of contents
1. [Environment setup](#environment-setup)
2. [How to run](#how-to-run)
## Environment setup
Install dependencies:
```shell
cd Anti-DreamBooth
conda create -n anti-dreambooth python=3.9  
conda activate anti-dreambooth  
pip install -r requirements.txt  
```
Pretrained stable diffusion model v2.1 is used.Please [download it](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) and put it under `./stable-diffusion/`.
## How to run
Run default ASPL attack:
```bash
bash scripts/attack_with_aspl.sh
```
It will first generate perturbed images with ASPL attack and train the dreambooth model on them. After training, the model will generate the instance images.

Run ASPL_REM attack:
```bash
bash scripts/aspl_REM.sh
```
It will generate perturbed images with ASPL_REM attack.

Run adversarial defense to ASPL attack:
```bash
bash scripts/aspl_adv.sh
```

Run adversarial defense to ASPL_REM attack:
```bash
bash scripts/aspl_REM_adv.sh
```
