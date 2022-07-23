"""REPredictor."""
import logging
import os
import pickle

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from repredictor.predictor.repredictor.network import REPredictorNetwork
from repredictor.utils.functional import get_value

import torch

from repredictor.data import BasicDataset
from repredictor.predictor.base.basic_predictor import BasicPredictor
from repredictor.predictor.repredictor.preprocessor import RePredictorPreprocessor

CORE_ROLES = [
    1,  # :ARG1
    2,  # :ARG0
    3,  # :ARG2
]


def _pad_zero(seq: list[int], seq_len: int) -> list[int]:
    """Pad zero to sequence.

    Args:
        seq (list[int]): original sequence.
        seq_len (int): target length.

    Returns:
        list[int]: padded sequence.
    """
    if len(seq) < seq_len:
        seq = seq + [0] * (seq_len - len(seq))
    return seq


class REDataset(BasicDataset):
    """Chain model dataset."""

    def __init__(self, data, num_args=25, rich_event=True):
        super(REDataset, self).__init__(data)
        self._data = data
        self._num_args = num_args
        self._rich_event = rich_event

    def convert_event(self,
                      event: tuple[int, int, list[int], list[int], list[int]]
                      ) -> list[int]:
        """Convert event into vector.

        Args:
            event (tuple): the indexed event.

        Returns:
            list[int]: a single event vector.
        """
        verb, role, roles, values, concepts = event
        if not self._rich_event:
            rest_roles = [(r, v, c) for r, v, c in zip(roles, values, concepts)
                          if r in CORE_ROLES]
            if len(rest_roles) > 0:
                roles, values, concepts = zip(*rest_roles)
                roles, values, concepts = list(roles), list(values), list(concepts)
            else:
                roles, values, concepts = [], [], []
        roles = _pad_zero(roles, self._num_args)
        values = _pad_zero(values, self._num_args)
        concepts = _pad_zero(concepts, self._num_args)
        return [verb, role] + roles + values + concepts

    def __getitem__(self, item):
        context, choices, target = self._data[item]
        context = [self.convert_event(e) for e in context]
        choices = [self.convert_event(e) for e in choices]
        context, choices, target = tuple(
            map(torch.tensor, (context, choices, target)))
        return context, choices, target

    @classmethod
    def from_file(cls, fp, num_args=25, rich_event=True):
        """Load dataset from file."""
        with open(fp, "rb") as f:
            data = pickle.load(f)
        return cls(data, num_args, rich_event)


class REPredictor(BasicPredictor):
    """Rich event predictor."""

    def __init__(self,
                 config: dict,
                 device: str = "cpu"):
        """Construction method of BasicPredictor class.

        Args:
            config (dict): configuration.
            device (str, optional): device name.
                Defaults to "cpu".
        """
        self._logger = logging.getLogger("repredictor.REPredictor")
        # Preprocess
        self._preprocessor = RePredictorPreprocessor(config)
        self._preprocess_dir = self._preprocessor.preprocess_dir
        # Hyper-parameters
        self._role_size = config["model"]["role_size"]
        self._concept_size = config["model"]["concept_size"]
        self._rich_event = config["model"]["rich_event"]
        self._num_args = config["model"]["num_args"]
        self._use_concept = config["model"]["use_concept"]
        self._hidden_dim_event = get_value(config["model"], "hidden_dim_event")
        self._num_layers_event = get_value(config["model"], "num_layers_event")
        self._num_heads_event = get_value(config["model"], "num_heads_event")
        self._dim_feedforward_event = get_value(config["model"], "dim_feedforward_event")
        self._num_layers_seq = config["model"]["num_layers_seq"]
        self._num_heads_seq = config["model"]["num_heads_seq"]
        self._dim_feedforward_seq = config["model"]["dim_feedforward_seq"]
        self._event_func = config["model"]["event_func"]
        self._attention_func = config["model"]["attention_func"]
        self._score_func = config["model"]["score_func"]
        super(REPredictor, self).__init__(config, device)

    def build_model(self):
        """Build repredictor model."""
        if not os.path.exists(self._model_dir):
            os.makedirs(self._model_dir)
        self._model = REPredictorNetwork(
            vocab_size=self._vocab_size,
            role_size=self._role_size,
            concept_size=self._concept_size,
            embedding_dim=self._embedding_dim,
            event_dim=self._event_dim,
            seq_len=self._seq_len,
            dropout=self._dropout,
            hidden_dim_event=self._hidden_dim_event,
            num_layers_event=self._num_layers_event,
            num_heads_event=self._num_heads_event,
            dim_feedforward_event=self._dim_feedforward_event,
            num_layers_seq=self._num_layers_seq,
            num_heads_seq=self._num_heads_seq,
            dim_feedforward_seq=self._dim_feedforward_seq,
            event_func=self._event_func,
            attention_func=self._attention_func,
            score_func=self._score_func,
            num_args=self._num_args,
            rich_event=self._rich_event,
            use_concept=self._use_concept)
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
        dev_set = REDataset.from_file(dev_fp, self._num_args, self._use_concept)
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
        # Train
        best_performance = 0.
        for epoch in range(npoch):
            logger.info(f"Epoch: {epoch}")
            loss_buffer = []
            for train_fn in os.listdir(train_fp):
                train_path = os.path.join(train_fp, train_fn)
                train_set = REDataset.from_file(train_path, self._num_args, self._use_concept)
                train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
                batch_num = 0
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
                    # evaluate
                    batch_num += 1
                    if batch_num % interval == 0:
                        cur_performance = self.validate(
                            valid_set=dev_set,
                            verbose=True)
                        if cur_performance > best_performance:
                            self.save_checkpoint("best", verbose=False)
                            best_performance = cur_performance
                            if verbose:
                                self._logger.info(
                                    f"Accuracy on dev: {best_performance:.2%}")
                cur_performance = self.validate(
                    valid_set=dev_set,
                    verbose=True)
                if cur_performance > best_performance:
                    self.save_checkpoint("best", verbose=False)
                    best_performance = cur_performance
                    if verbose:
                        self._logger.info(
                            f"Iter: {batch_num}, "
                            f"Accuracy on dev: {best_performance:.2%}")
            logger.info(f"Average loss: {sum(loss_buffer) / len(loss_buffer):.4f}")
            logger.info(f"Best accuracy on dev: {best_performance:.2%}")

    def validate(self,
                 valid_set: BasicDataset = None,
                 valid_fp: str = None,
                 verbose: bool = False) -> float:
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
            float: performance under evaluation metric.
        """
        if valid_set is None:
            if valid_fp is None:
                valid_fp = os.path.join(self._preprocess_dir, "dev_idx.pkl")
            valid_set = REDataset.from_file(valid_fp, self._num_args, self._use_concept)
        predict_label, true_label = self.predict(pred_set=valid_set)
        accuracy = self.evaluate(predict_label, true_label)
        return accuracy

    def test(self,
             test_set: BasicDataset = None,
             test_fp: str = None,
             verbose: bool = True) -> float:
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
            float: performance under evaluation metric.
        """
        if test_set is None:
            if test_fp is None:
                test_fp = os.path.join(self._preprocess_dir, "test_idx.pkl")
            test_set = REDataset.from_file(test_fp, self._num_args, self._use_concept)
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
            pred_set = REDataset.from_file(pred_fp, self._num_args, self._use_concept)
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

