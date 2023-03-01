# RePredictor
Experiment code for:

Long Bai, Saiping Guan, Zixuan Li, Jiafeng Guo, Xiaolong Jin, Xueqi Cheng.
"*Rich Event Modeling for Script Event Prediction*", AAAI 2023


## Install

Install this package via:

```bash 
pip install .
```

Code for generating MCNC-rich is not fully prepared yet.

## Preprocess

```bash 
python scripts/preprocess.py --config <ConfigFile>
```

## Train

```bash 
python scripts/train.py --config <ConfigFile> --device <Device>
```

## Test

```bash 
python scripts/test.py --config <ConfigFile> --device <Device>
```
