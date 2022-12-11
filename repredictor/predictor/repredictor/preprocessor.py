"""Preprocessor for chain model."""

import logging
import os
import pickle
from functools import partial
from typing import Tuple, Dict, List

import bs4

from repredictor.predictor.base.basic_preprocessor import BasicPreprocessor
from repredictor.utils.document import Document
from repredictor.utils.entity import Entity
from repredictor.utils.event import Event
from repredictor.utils.functional import lookup


def _indexing_event(event: Tuple[str, str, List[str], List[str], List[str]],
                    word_dict: Dict[str, int],
                    role_dict: Dict[str, int],
                    concept_dict: Dict[str, int]
                    ) -> Tuple[int, int, List[int], List[int], List[int]]:
    """Indexing an event.

    Args:
        event (tuple): event representation contain verb, role, and arguments.
        word_dict (dict[str, int]): word dictionary.
        role_dict (dict[str, int]): role dictionary.
        concept_dict (dict[str, int]): type dictionary.

    Returns:
        tuple: the indexed event.
    """
    verb, role, roles, values, concepts = event
    verb_i = lookup(verb, word_dict)
    role_i = lookup(role, role_dict)
    map_func_role = partial(lookup, dictionary=role_dict)
    map_func_word = partial(lookup, dictionary=word_dict)
    map_func_concept = partial(lookup, dictionary=concept_dict)
    roles_i = list(map(map_func_role, roles))
    values_i = list(map(map_func_word, values))
    concepts_i = list(map(map_func_concept, concepts))
    return verb_i, role_i, roles_i, values_i, concepts_i


class RePredictorPreprocessor(BasicPreprocessor):
    """Chain model preprocessor."""

    def __init__(self, config):
        """Construction method for RePredictorPreprocessor class.

        Args:
            config (dict): configuration.
        """
        super(RePredictorPreprocessor, self).__init__(config)
        self._logger = logging.getLogger("repredictor.RePredictor.preprocessor")

    def load_frame2verb(self):
        """Load frame2verb mapping."""
        self.load_word_dict()
        pb_dir = self._config["pb_dir"]
        pb_dir = os.path.join(pb_dir, "frames")
        xmls = [fn for fn in os.listdir(pb_dir) if fn.endswith(".xml")]
        frame2verb = {}
        lemmaid = {}
        word_dict = self._word_dict
        for fn in xmls:
            fp = os.path.join(pb_dir, fn)
            with open(fp, "r") as f:
                dom = bs4.BeautifulSoup(f.read(), "lxml")
            for predicate in dom.findAll("predicate"):
                lemma = predicate.get("lemma")
                lemmaid.setdefault(lemma, 0)
                for roleset in predicate.findAll("roleset"):
                    frame = roleset.get("id")
                    frameid = lookup(frame, word_dict)
                    if frameid != 0:
                        if lemmaid[lemma] == 0:
                            lemmaid[lemma] = frameid
                        else:
                            lemmaid[lemma] = min(lemmaid[lemma], frameid)
                    frame2verb.setdefault(frame, lemma)
        frameid2verbid = {}
        for frame, lemma in frame2verb.items():
            frameid = lookup(frame, word_dict)
            frameid2verbid.setdefault(frameid, lemmaid[lemma])
        return frameid2verbid

    def generate_a_question(self,
                            entity: Entity,
                            context: List[Event],
                            choices: List[Event],
                            target: int,
                            doc: Document):
        """Generate an input question.

        Args:
            entity (Entity): the protagonist.
            context (list[Event]): the context events.
            choices (list[Event]): the choices events.
            target (int): the answer index.
            doc (Document): the document.

        Returns:
            the formatted question.
        """
        doc_id = doc.doc_id
        entities = doc.entities
        context = [_.rich_repr(entity, entities) for _ in context]
        choices = [_.rich_repr(entity, entities) for _ in choices]
        return context, choices, target

    def indexing_a_question(self, question):
        """Indexing a question.

        Args:
            question: the elements needed for question.

        Returns:
            the indexed question.
        """
        context, choices, target = question
        word_dict = self._word_dict
        role_dict = self._role_dict
        concept_dict = self._concept_dict
        map_func = partial(
            _indexing_event,
            word_dict=word_dict,
            role_dict=role_dict,
            concept_dict=concept_dict)
        context_idx = list(map(map_func, context))
        choices_idx = list(map(map_func, choices))
        return context_idx, choices_idx, target

    def dataset_statistics(self) -> None:
        """Statistics on datasets."""
        preprocess_dir = self._preprocess_dir
        # Count words
        if self._word_dict is None:
            self.load_word_dict()
        self._logger.info(f"Totally {len(self._word_dict)} words in word_dict.")
        # Count roles
        if self._role_dict is None:
            self.load_role_dict()
        self._logger.info(f"Totally {len(self._role_dict)} words in role dict.")
        # Count concepts
        if self._concept_dict is None:
            self.load_concept_dict()
        self._logger.info(f"Totally {len(self._concept_dict)} words in concept dict.")
        # Count train question indices
        train_idx_dir = os.path.join(preprocess_dir, "train_idx")
        num_train_questions = 0
        num_args = 0
        total_args = 0
        total_events = 0
        for fn in os.listdir(train_idx_dir):
            train_idx_path = os.path.join(train_idx_dir, fn)
            with open(train_idx_path, "rb") as f:
                train_idx = pickle.load(f)
            for context, choices, target in train_idx:
                for event in context + choices:
                    num_args = max(num_args, len(event[2]))
                    total_args += len(event[2])
                    total_events += 1
            num_train_questions += len(train_idx)
        self._logger.info(f"Totally {num_train_questions} questions in train set.")
        self._logger.info(f"Up to {num_args} args for each event.")
        self._logger.info(f"Averagely {total_args/total_events:.2f} args for each event.")
        # Count dev question indices
        dev_idx_path = os.path.join(preprocess_dir, "dev_idx.pkl")
        with open(dev_idx_path, "rb") as f:
            dev_idx = pickle.load(f)
        self._logger.info(f"Totally {len(dev_idx)} questions in dev set.")
        # Count test question indices
        test_idx_path = os.path.join(preprocess_dir, "test_idx.pkl")
        with open(test_idx_path, "rb") as f:
            test_idx = pickle.load(f)
        self._logger.info(f"Totally {len(test_idx)} questions in test set.")
