"""REPredictor."""
import logging
import os
import pickle
from typing import Tuple, List

from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiplicativeLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def _pad_zero(seq: List[int], seq_len: int) -> List[int]:
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

    def __init__(self, data, num_args: int = 23,
                 rich_event: bool = True,
                 use_frame: bool = True,
                 frame2verb: dict = None):
        """Construction method for REDataset.

        Args:
            data: data.
            num_args (int): max number of arguments.
                Defaults to 23.
            rich_event (bool): whether to use non-core arguments.
                Defaults to True.
            use_frame (bool): use frame instead of verb.
                Defaults to True.
            frame2verb (dict): map frame to verb lemma.
        """
        super(REDataset, self).__init__(data)
        self._data = data
        self._num_args = num_args
        self._rich_event = rich_event
        self._use_frame = use_frame
        self.frame2verb = frame2verb

    def convert_event(self,
                      event: Tuple[int, int, List[int], List[int], List[int]]
                      ) -> List[int]:
        """Convert event into vector.

        Args:
            event (tuple): the indexed event.

        Returns:
            list[int]: a single event vector.
        """
        verb, role, roles, values, concepts = event
        if not self._rich_event:
            # Filter arguments according to role type
            rest_roles = [(r, v, c) for r, v, c in zip(roles, values, concepts)
                          if r in CORE_ROLES]
            if len(rest_roles) > 0:
                # Split fields
                roles, values, concepts = map(list, zip(*rest_roles))
            else:
                roles, values, concepts = [], [], []
        if not self._use_frame:
            if verb in self.frame2verb:
                verb = self.frame2verb[verb]
            else:
                verb = 0
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
    def from_file(cls, fp: str, num_args: int = 25, rich_event: bool = True,
                  use_frame: bool = True, frame2verb: dict = None):
        """Load dataset from file.

        Args:
            fp (str): file path.
            num_args (int): max number of arguments.
                Defaults to 25.
            rich_event (bool): whether to use non-core arguments.
                Defaults to True.
            use_frame (bool)
            frame2verb (dict)
        """
        with open(fp, "rb") as f:
            data = pickle.load(f)
        return cls(data, num_args, rich_event, use_frame, frame2verb)


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
        # Hyper-parameters
        self._role_size = config["model"]["role_size"]
        self._concept_size = config["model"]["concept_size"]
        self._rich_event = config["model"]["rich_event"]
        self._num_args = config["model"]["num_args"]
        self._use_concept = config["model"]["use_concept"]
        self._use_frame = get_value(config["model"], "use_frame", True)
        self._concept_only = get_value(config["model"], "concept_only", False)
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
        self.frame2verb = None
        super(REPredictor, self).__init__(config, device)

    def build_model(self):
        """Build repredictor model."""
        # Preprocess
        self._preprocessor = RePredictorPreprocessor(self._config)
        self._preprocess_dir = self._preprocessor.preprocess_dir
        self.frame2verb = self._preprocessor.load_frame2verb()
        self._logger = logging.getLogger("repredictor.REPredictor")
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
            use_concept=self._use_concept,
            concept_only=self._concept_only)
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
        dev_set = REDataset.from_file(
            dev_fp, self._num_args, self._use_concept,
            self._use_frame, self.frame2verb)
        # Prepare model and optimizer
        model = self._model
        self._logger.info(f"Model structure:\n{model}")
        optimizer = self._optimizer
        scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda x: 0.8, verbose=True)
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
                train_set = REDataset.from_file(
                    train_path, self._num_args, self._use_concept, self._use_frame, self.frame2verb)
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
            # scheduler.step()
            # scheduler.step(best_performance)

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
            valid_set = REDataset.from_file(
                valid_fp, self._num_args, self._use_concept, self._use_frame, self.frame2verb)
        predict_label, true_label = self.predict(pred_set=valid_set)
        accuracy = self.evaluate(predict_label, true_label)
        if verbose:
            self._logger.info(f"Accuracy on dev: {accuracy:.2%}")
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
            test_set = REDataset.from_file(
                test_fp, self._num_args, self._use_concept, self._use_frame, self.frame2verb)
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
            pred_set (dataset, optional): dataset to be predicted.
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
            pred_set = REDataset.from_file(
                pred_fp, self._num_args, self._use_concept, self._use_frame, self.frame2verb)
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
            pred_set = REDataset.from_file(
                pred_fp, self._num_args, self._use_concept, self._use_frame, self.frame2verb)
        # predict
        model = self._model
        model.eval()
        device = self._device
        batch_size = self._batch_size
        pred_loader = DataLoader(pred_set, batch_size=batch_size)
        num_args = []
        attn_matrices = []
        predict_label, true_label = [], []
        with torch.no_grad():
            for context, choices, target in pred_loader:
                context = context.to(device)
                choices = choices.to(device)
                target = target.to(device)
                pred = model(context, choices)
                attn = model.get_event_attention_matrix(context)
                predict_label.append(pred.argmax(1).cpu())
                true_label.append(target.cpu())
                attn_matrices.append(attn.cpu())
                num_args.append(model.get_actual_num_args(context).cpu())
        predict_label = torch.cat(predict_label, dim=0)
        true_label = torch.cat(true_label, dim=0)
        attn = torch.cat(attn_matrices, dim=0)
        num_args = torch.cat(num_args, dim=0)
        results = {
            "predict_label": predict_label.numpy(),
            "true_label": true_label.numpy(),
            "attn": attn.numpy(),
            "num_args": num_args.numpy(),
            "accuracy": self.evaluate(predict_label, true_label)
        }
        self.save_results(results)
        return results
