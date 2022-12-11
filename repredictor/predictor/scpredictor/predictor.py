"""SCPredictor."""
import logging
import os

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from repredictor.data import BasicDataset
from repredictor.predictor.base.basic_predictor import BasicPredictor
from repredictor.predictor.scpredictor.network import SCPredictorNetwork
from repredictor.predictor.scpredictor.preprocessor import ScpredictorPreprocessor


class SCDataset(BasicDataset):
    """Dataset for SCPredictor."""

    def __init__(self, data):
        """Construction method for BasicDataset.

        Args:
            data: data
        """
        super(SCDataset, self).__init__(data)

    def __getitem__(self, idx: int):
        """Get input and output.

        Args:
            idx (int): index of the sample.

        Returns:
            Any: input.
            Any: output.
        """
        context, choices, target = tuple(map(torch.tensor, self._data[idx]))
        return context, choices, target


class SCPredictor(BasicPredictor):
    """Chain predictor class."""

    def __init__(self, config: dict, device: str = "cpu"):
        """Construction method of SCPredictor class.

        Args:
            config (dict): configuration.
            device (str, optional): device name.
                Defaults to "cpu".
        """
        self._num_layers = config["model"]["num_layers"]
        self._num_heads = config["model"]["num_heads"]
        self._dim_feedforward = config["model"]["dim_feedforward"]
        self._score_func = config["model"]["score_func"]
        self._attention_func = config["model"]["attention_func"]
        super(SCPredictor, self).__init__(config, device)

    def build_model(self):
        """Build scpredictor model."""
        self._preprocessor = ScpredictorPreprocessor(self._config)
        self._preprocess_dir = self._preprocessor.preprocess_dir
        self._logger = logging.getLogger("repredictor.SCPredictor")
        if not os.path.exists(self._model_dir):
            os.makedirs(self._model_dir)
        self._model = SCPredictorNetwork(
            vocab_size=self._vocab_size,
            embedding_dim=self._embedding_dim,
            event_dim=self._event_dim,
            seq_len=self._seq_len,
            dropout=self._dropout,
            num_layers=self._num_layers,
            num_heads=self._num_heads,
            dim_feedforward=self._dim_feedforward,
            score_func=self._score_func,
            attention_func=self._attention_func)
        self._model.to(self._device)
        lr = self._lr
        weight_decay = self._weight_decay
        self._optimizer = Adam(
            params=self._model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self,
              train_set: BasicDataset = None,
              train_fp: str = None,
              dev_set: BasicDataset = None,
              dev_fp: str = None,
              verbose: bool = True) -> None:
        """Train the model on train set.

        Args:
            train_set (BasicDataset, optional): train set.
                Defaults to None.
            train_fp (str, optional): train file path.
                In many cases, train set is too large to fully load into memory.
                Then, this argument should be a directory.
                Only used when train_set is None.
                Defaults to None.
            dev_set (BasicDataset, optional): dev set.
                Defaults to None.
            dev_fp (str, optional): dev file path.
                Defaults to None
            verbose (bool): whether to print training details in logger.
                Defaults to True.
        """
        # Load index data
        preprocess_dir = self._preprocess_dir
        if train_fp is None:
            train_fp = os.path.join(preprocess_dir, "train_idx")
        if dev_fp is None:
            dev_fp = os.path.join(preprocess_dir, "dev_idx.pkl")
        dev_set = SCDataset.from_file(dev_fp)
        # Prepare model and optimizer
        model = self._model
        optimizer = self._optimizer
        loss_fn = nn.CrossEntropyLoss()
        # Prepare data and parameters
        device = self._device
        batch_size = self._batch_size
        npoch = self._npoch
        interval = self._interval
        logger = self._logger
        pbar = None
        # Train
        best_performance = 0.
        for epoch in range(npoch):
            logger.info(f"Epoch: {epoch}")
            loss_buffer = []
            for train_fn in os.listdir(train_fp):
                train_path = os.path.join(train_fp, train_fn)
                train_set = SCDataset.from_file(train_path)
                train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
                batch_num = 0
                if self._progress_bar:
                    pbar = tqdm(total=len(train_set), ascii=True)
                for context, choices, target in train_loader:
                    # forward
                    model.train()
                    context = context.to(device)
                    choices = choices.to(device)
                    target = target.to(device)
                    pred = model(context, choices)
                    loss = loss_fn(pred, target)
                    loss_buffer.append(loss.item())
                    # backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if pbar is not None:
                        pbar.update(len(target))
                        pbar.set_description(f"Loss: {loss.item():.4f} "
                                             f"Best dev: {best_performance:.2%}")
                    # evaluate
                    batch_num += 1
                    if batch_num % interval == 0:
                        cur_performance = self.validate(
                            valid_set=dev_set,
                            verbose=False)
                        best_performance = self.compare_and_update(
                            cur_perf=cur_performance,
                            best_perf=best_performance,
                            verbose=False)
                if pbar is not None:
                    pbar.close()
            cur_performance = self.validate(
                valid_set=dev_set,
                verbose=False)
            best_performance = self.compare_and_update(
                cur_perf=cur_performance,
                best_perf=best_performance,
                verbose=False)
            logger.info(f"Average loss: {sum(loss_buffer) / len(loss_buffer):.4f}")
            logger.info(f"Best accuracy on dev: {best_performance:.2%}")

    def validate(self,
                 valid_set: BasicDataset = None,
                 valid_fp: str = None,
                 verbose: bool = False) -> dict or float:
        """Evaluate the model on dev/valid set.

        Args:
            valid_set (BasicDataset, optional): dev/valid set.
                Defaults to None.
            valid_fp (str, optional): the dev/valid file path.
                Only used when valid_set is None.
                Defaults to None.
            verbose (bool): whether to print evaluation details in logger.
                Defaults to False.

        Returns:
            float or dict: performance under evaluation metric.
        """
        if valid_set is None:
            if valid_fp is None:
                valid_fp = os.path.join(self._preprocess_dir, "dev_idx.pkl")
            valid_set = SCDataset.from_file(valid_fp)
        predict_label, true_label = self.predict(pred_set=valid_set)
        accuracy = self.evaluate(predict_label, true_label)
        if verbose:
            self._logger.info(f"Accuracy on dev: {accuracy:.2%}")
        return accuracy

    def test(self,
             test_set: BasicDataset = None,
             test_fp: str = None,
             verbose: bool = True) -> dict or float:
        """Evaluate the model on test set.

        Args:
            test_set (dataset, optional): test set.
                Defaults to None.
            test_fp (str, optional): test file path.
                Only used when test_set is None.
                Defaults to None.
            verbose (bool): whether to print evaluation details in logger.
                Defaults to True.

        Returns:
            float or dict: performance under evaluation metric.
        """
        if test_set is None:
            if test_fp is None:
                test_fp = os.path.join(self._preprocess_dir, "test_idx.pkl")
            test_set = SCDataset.from_file(test_fp)
        predict_label, true_label = self.predict(pred_set=test_set)
        accuracy = self.evaluate(predict_label, true_label)
        if verbose:
            self._logger.info(f"Accuracy on test: {accuracy:.2%}")
        return accuracy

    def predict(self,
                pred_set: BasicDataset = None,
                pred_fp: str = None):
        """Predict samples.

        Args:
            pred_set (dataset, optional): dataset to be predict.
                Defaults to None.
            pred_fp (str, optional): dataset file path.
                Only used when pred_set is None.
                Defaults to None.

        Returns:
            Any: results.
        """
        preprocess_dir = self._preprocess_dir
        if pred_set is None:
            if pred_fp is None:
                pred_fp = os.path.join(preprocess_dir, "test_idx.pkl")
            pred_set = SCDataset.from_file(pred_fp)
        # predict
        model = self._model
        model.eval()
        device = self._device
        batch_size = self._batch_size
        pred_loader = DataLoader(pred_set, batch_size=batch_size)
        predict_label, true_label = [], []
        with torch.no_grad():
            for context, choices, target in pred_loader:
                context = context.to(device)
                choices = choices.to(device)
                target = target.to(device)
                pred = model(context, choices)
                predict_label.append(pred.argmax(1).cpu())
                true_label.append(target.cpu())
        predict_label = torch.cat(predict_label, dim=0)
        true_label = torch.cat(true_label, dim=0)
        return predict_label, true_label

    def analyze(self,
                pred_set: BasicDataset = None,
                pred_fp: str = None):
        """Analyze the results.

        Args:
            pred_set (dataset, optional): dataset to be predicted.
                Defaults to None.
            pred_fp (str, optional): dataset file path.
                Only used when pred_set is None.
                Defaults to None.
        """
        preprocess_dir = self._preprocess_dir
        if pred_set is None:
            if pred_fp is None:
                pred_fp = os.path.join(preprocess_dir, "dev_idx.pkl")
            pred_set = SCDataset.from_file(pred_fp)
        # predict
        model = self._model
        model.eval()
        device = self._device
        batch_size = self._batch_size
        pred_loader = DataLoader(pred_set, batch_size=batch_size)
        predict_label, true_label = [], []
        with torch.no_grad():
            for context, choices, target in pred_loader:
                context = context.to(device)
                choices = choices.to(device)
                target = target.to(device)
                pred = model(context, choices)
                predict_label.append(pred.argmax(1).cpu())
                true_label.append(target.cpu())
        predict_label = torch.cat(predict_label, dim=0)
        true_label = torch.cat(true_label, dim=0)
        results = {
            "predict_label": predict_label.numpy(),
            "true_label": true_label.numpy(),
            "accuracy": self.evaluate(predict_label, true_label)
        }
        self.save_results(results)
        return results
