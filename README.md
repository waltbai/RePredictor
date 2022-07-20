# Template for Neural Network Models

Change ``package_name`` to your own module name.

## Install

Install this package via:

```bash 
pip install .
```

## Implement Predictor

Methods must override:
```python
from package_name.models import BasicPredictor

class MyPredictor(BasicPredictor):
    def __init__(self, config, device="cpu"):
        # Your own settings here
        super(MyPredictor, self).__init__(config, device)
    
    def build_model(self):
        # Construct your own model
        pass

    def train(self,
              train_set: BasicDataset = None,
              train_fp: str = None,
              verbose: bool = True) -> None:
        # Implement train process
        pass

    def validate(self,
                 valid_set: BasicDataset = None,
                 valid_fp: str = None,
                 verbose: bool = False) -> dict or float:
        # Implement validate process
        # It is recommended to use predict and evaluate method
        pass

    def test(self,
             test_set: BasicDataset = None,
             test_fp: str = None,
             verbose: bool = True) -> dict or float:
        # Implement test process
        # It is recommended to use predict and evaluate method
        pass

    def predict(self,
                pred_set: BasicDataset = None,
                pred_fp: str = None):
        # Implement predict process
        pass

    def evaluate(self, pred_label, true_label) -> dict or float:
        # Implement evaluate process
        pass
```

If you need another abstract class:
```python
from abc import ABC, abstractmethod
from package_name.models import BasicPredictor

class MyAbstractPredictor(BasicPredictor):
    def __init__(self, config, device="cpu"):
        # Your own settings here
        super(MyPredictor, self).__init__(config, device)
    
    # Skip the abstract methods that need further implement
    def evaluate(self, pred_label, true_label) -> dict or float:
        # Implement evaluate process
        pass
```

## Implement Dataset

Methods must override:
```python
from package_name.models import BasicDataset

class MyDataset(BasicDataset):
    def __init__(self, data, train=False):
        super(MyDataset, self).__init__(data, train)
    
    def get_input(self, idx: int):
        # Implement get input
        pass

    def get_input_and_output(self, idx: int):
        # Implement get input and output
        pass
```

In most cases, you can use torch `DataLoader`. However, in some special cases,
such as using graph as input, you may construct batches by yourself. Then you
can inherit `BasicDataLoader` to design your own dataloader:
```python
from package_name.models import BasicDataLoader

class MyDataLoader(BasicDataLoader):
    def __init__(self,
                 dataset: BasicDataset,
                 batch_size: int = 1,
                 shuffle: bool = False):
        super(MyDataLoader, self).__init__(dataset, batch_size, shuffle)
    
    def construct_batch(self, indices: list):
        # Implement your batch construction strategies here.
        pass
```

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
