"""SCPredictor."""
import logging
import math
import os
import pickle
from collections import Counter
from typing import Tuple

import numpy
import torch
from tqdm import tqdm

from repredictor.data import BasicDataset
from repredictor.predictor.base.basic_predictor import BasicPredictor
from repredictor.predictor.pmi.preprocessor import PmiPreprocessor
from repredictor.utils import load_config, save_config


class PgrDataset(BasicDataset):
    """Predicate gr dataset."""

    def __init__(self, data, train: bool = True):
        super(PgrDataset, self).__init__(data)
        self._train = train

    def __getitem__(self, item):
        return self._data[item]

    @classmethod
    def from_file(cls, fp: str, train: bool = True):
        """Load data from file.

        Args:
            fp (str): file path
            train (bool): train mode.
                Defaults to True.
        """
        with open(fp, "rb") as f:
            data = pickle.load(f)
        return cls(data, train)


class PmiPredictor(BasicPredictor):
    """Pmi predictor class."""

    def __init__(self, config: dict, device: str = "cpu"):
        """Construction method of SCPredictor class.

        Args:
            config (dict): configuration.
            device (str, optional): device name.
                Defaults to "cpu".
        """
        self._event_count = None
        self._pair_count = None
        self._total_events = None
        self._total_pairs = None
        # unordered model outperforms ordered on about 1% on accuracy
        self._ordered = config["model"]["ordered"]
        self._event_threshold = config["model"]["event_threshold"]
        self._pair_threshold = config["model"]["pair_threshold"]
        super(PmiPredictor, self).__init__(config, device)

    def build_model(self):
        """Build scpredictor model."""
        # logger
        self._logger = logging.getLogger("repredictor.PMIPredictor")
        # Preprocess
        self._preprocessor = PmiPreprocessor(self._config)
        self._preprocess_dir = self._preprocessor.preprocess_dir
        # Model
        self._event_count = Counter()
        self._pair_count = Counter()
        self._total_events = sum(self._event_count.values())
        self._total_pairs = sum(self._pair_count.values())

    @classmethod
    def from_checkpoint(cls,
                        checkpoint_dir: str,
                        device: str = "cpu",
                        verbose: bool = True):
        """Load model from checkpoint.

        Args:
            checkpoint_dir (str): checkpoint directory
            device (str, optional): the computing device.
                Defaults to "cpu".
            verbose (bool, optional): whether to print details in logger.
                Defaults to True.

        Returns:
            PmiPredictor: the trained model
        """
        config_path = os.path.join(checkpoint_dir, "config.yml")
        config = load_config(config_path)
        predictor = cls(config, device)
        if verbose:
            predictor._logger.info(
                f"Loading checkpoint from {checkpoint_dir}.")
        model_path = os.path.join(checkpoint_dir, "model.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        predictor._event_count = Counter(model["event_count"])
        predictor._pair_count = Counter(model["pair_count"])
        predictor._total_events = sum(predictor._event_count.values())
        predictor._total_pairs = sum(predictor._pair_count.values())
        return predictor

    def save_checkpoint(self,
                        checkpoint: str,
                        verbose: bool = True) -> None:
        """Save checkpoint.

        Args:
            checkpoint (str): unique name to specify a checkpoint.
            verbose (bool, optional): whether to print details in logger.
                Defaults to True.
        """
        checkpoint_dir = os.path.join(self._model_dir, checkpoint)
        # Create directory for checkpoint
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save config file
        config_path = os.path.join(checkpoint_dir, "config.yml")
        save_config(self._config, config_path)
        # Save model parameters
        model_path = os.path.join(checkpoint_dir, "model.pkl")
        model = {
            "event_count": dict(self._event_count),
            "pair_count": dict(self._pair_count)
        }
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        # logging
        if verbose:
            self._logger.info(
                f"Save checkpoint {checkpoint} to {checkpoint_dir}.")

    def pmi(self,
            e1: Tuple[str, str],
            e2: Tuple[str, str]) -> float:
        """Count PMI for an event pair.

        Basically the same to Granroth-Wilding's CandjNarrativeChain model.

        Args:
            e1 (Tuple[str, str]): (verb, dep) tuple.
            e2 (Tuple[str, str]): (verb, dep) tuple.

        Returns:
            float: pmi score for event pair.
        """
        # If in unordered mode, (e1, e2) and (e2, e1) should be the same pair.
        if not self._ordered and e1 > e2:
            e1, e2 = e2, e1
        raw_count1 = self._event_count[e1]
        raw_count2 = self._event_count[e2]
        # One of the event is never seen before
        if raw_count1 == 0 or raw_count2 == 0:
            return 0.
        pair_count = self._pair_count[(e1, e2)]
        # The event pair is never seen before
        if pair_count == 0:
            return 0.
        # Compute pmi
        log_e1_count = math.log(raw_count1)
        log_e2_count = math.log(raw_count2)
        log_pair_count = math.log(pair_count)
        log_total_pairs = math.log(self._total_pairs)
        log_total_events = math.log(self._total_events)
        pmi = log_pair_count - log_total_pairs - log_e1_count - log_e2_count + \
            2 * log_total_events
        # Zero out negative PMIs
        if pmi <= 0.:
            return 0.
        # Discount score
        discount_min = float(min(raw_count1, raw_count2))
        discount = pair_count / (pair_count + 1.) * discount_min / (discount_min + 1.)
        return discount * pmi

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
            dev_set (BasicDataset, optional): not used.
            dev_fp (str, optional): not used.
            verbose (bool): whether to print training details in logger.
                Defaults to True.
        """
        # Load index data
        preprocess_dir = self._preprocess_dir
        if train_set is None:
            if train_fp is None:
                train_fp = os.path.join(preprocess_dir, "train.pkl")
            train_set = PgrDataset.from_file(train_fp, True)
        pbar = None
        self._logger.info("Counting event pairs ...")
        if self._progress_bar:
            pbar = tqdm(total=len(train_set), ascii=True)
            pbar.set_description("Processing chains")
        for i in range(len(train_set)):
            chain = train_set[i]
            # Update event count
            self._event_count.update(chain)
            # Update event pair
            pairs = []
            for e1, e2 in zip(chain, chain[1:]):
                # If in unordered mode,
                # (e1, e2) and (e2, e1) should be the same pair.
                if not self._ordered and e1 > e2:
                    e1, e2 = e2, e1
                pairs.append((e1, e2))
            self._pair_count.update(pairs)
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()
        # Remove low frequency events
        self._logger.info("Removing low frequency events ...")
        event_threshold = self._event_threshold
        to_remove = [v for v, f in self._event_count.most_common() if f < event_threshold]
        for key in to_remove:
            del self._event_count[key]
        # Remove event pairs
        self._logger.info("Removing low frequency pairs ...")
        to_remove = set(to_remove)
        pair_threshold = self._pair_threshold
        pairs_to_remove = [(e1, e2) for e1, e2 in self._pair_count.keys()
                           if e1 in to_remove or e2 in to_remove]
        for key in pairs_to_remove:
            del self._pair_count[key]
        pairs_to_remove = [v for v, f in self._pair_count.most_common() if f < pair_threshold]
        for key in pairs_to_remove:
            del self._pair_count[key]
        # Finish
        self._logger.info(f"Totally {len(self._event_count)} events, "
                          f"{len(self._pair_count)} pairs.")
        # Save model.
        self.save_checkpoint("best")

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
                valid_fp = os.path.join(self._preprocess_dir, "dev.pkl")
            valid_set = PgrDataset.from_file(valid_fp, False)
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
                test_fp = os.path.join(self._preprocess_dir, "test.pkl")
            test_set = PgrDataset.from_file(test_fp, False)
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
                pred_fp = os.path.join(preprocess_dir, "test.pkl")
            pred_set = PgrDataset.from_file(pred_fp, False)
        # predict
        predict_label = []
        true_label = []
        for i in range(len(pred_set)):
            context, choices, target = pred_set[i]
            scores = torch.zeros((len(context), len(choices)), dtype=torch.float)
            for context_idx, e1 in enumerate(context):
                for choices_idx, e2 in enumerate(choices):
                    scores[context_idx, choices_idx] = self.pmi(e1, e2)
            scores = scores.mean(dim=0)
            predict_label.append(int(torch.argmax(scores, dim=0)))
            true_label.append(target)
        predict_label = torch.tensor(predict_label)
        true_label = torch.tensor(true_label)
        return predict_label, true_label
