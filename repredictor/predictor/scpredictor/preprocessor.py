"""Preprocessor for scpredictor."""

import logging
import os
import pickle
from collections import Counter

from tqdm import tqdm

from repredictor.predictor.base.basic_preprocessor import BasicPreprocessor
from repredictor.utils.document import Document
from repredictor.utils.entity import Entity
from repredictor.utils.event import Event
from repredictor.utils.functional import lookup


def _indexing_event(event: tuple[str, str, str, str],
                    word_dict: dict,
                    role_dict: dict) -> tuple[int, int, int, int]:
    """Indexing an event.

    Args:
        event (tuple[str, str, str, str]): event representation.
        word_dict (dict): word dictionary.
        role_dict (dict): role dictionary.

    Returns:
        tuple[int, int, int, int]: indices of event
    """
    v, a0, a1, a2 = event
    v_i = lookup(v, word_dict)
    a0_i = lookup(a0, word_dict)
    a1_i = lookup(a1, word_dict)
    a2_i = lookup(a2, word_dict)
    # r = role_dict[r] if r in role_dict else role_dict["None"]
    return v_i, a0_i, a1_i, a2_i


class ScpredictorPreprocessor(BasicPreprocessor):
    """Preprocessor for scpredictor."""

    def __init__(self, config):
        """Construction method for ScpredictorPreprocessor class.

        Args:
            config (dict): configuration.
        """
        super(ScpredictorPreprocessor, self).__init__(config)
        self._logger = logging.getLogger("repredictor.SCPredictor.preprocessor")

    def generate_dictionaries(self,
                              train_dir: str,
                              min_count: int = 10,
                              overwrite: bool = False) -> None:
        """Generate word/role/type dictionary.

        Notice: scpredictor do not need role and type dictionary.

        Args:
            train_dir (str): train documents directory.
            min_count (int, optional): min count for words.
                Defaults to 10.
            overwrite (bool, optional): whether to overwrite old files.
                Defaults to False.
        """
        preprocess_dir = self._preprocess_dir
        word_dict_path = os.path.join(preprocess_dir, "word_dict.pkl")
        # Though generated, role and concept are not used in scpredictor
        role_dict_path = os.path.join(preprocess_dir, "role_dict.pkl")
        concept_dict_path = os.path.join(preprocess_dir, "concept_dict.pkl")
        exist_flag = os.path.exists(word_dict_path) and \
            os.path.exists(role_dict_path) and \
            os.path.exists(concept_dict_path)
        if exist_flag and not overwrite:
            self._logger.info(f"{word_dict_path}, "
                              f"{role_dict_path} and"
                              f"{concept_dict_path} already exists.")
        else:
            if self._stoplist is None:
                self.load_stoplist()
            stoplist = self._stoplist
            fn_list = os.listdir(train_dir)
            word_counter = Counter()
            self._logger.info(f"Generating dictionaries for {train_dir} ...")
            with tqdm(total=len(fn_list)) as pbar:
                pbar.set_description("Processed documents")
                for fn in fn_list:
                    fp = os.path.join(train_dir, fn)
                    doc = Document.from_file(fp)
                    for entity, chain in doc.get_chains(stoplist):
                        for event in chain:
                            words = event.quadruple(entity, doc.entities)
                            word_counter.update(words)
                    pbar.update(1)
            word_stat = word_counter.most_common()
            word_stat = [(w, f) for w, f in word_stat if f >= min_count]
            word_dict = {"None": 0}
            for w, f in word_stat:
                word_dict.setdefault(w, len(word_dict))
            role_dict = {"None": 0}
            concept_dict = {"None": 0}
            with open(word_dict_path, "wb") as f:
                pickle.dump(word_dict, f)
            with open(role_dict_path, "wb") as f:
                pickle.dump(role_dict, f)
            with open(concept_dict_path, "wb") as f:
                pickle.dump(concept_dict, f)
            self._word_dict = word_dict
            self._role_dict = role_dict
            self._concept_dict = concept_dict
            self._logger.info(f"Word dictionary save to {word_dict_path}, "
                              f"totally {len(word_dict)} words.")
            self._logger.info(f"Role dictionary save to {role_dict_path}, "
                              f"totally {len(role_dict)} roles.")
            self._logger.info(f"Concept dictionary save to {concept_dict_path}, "
                              f"totally {len(concept_dict)} concepts.")

    def generate_a_question(self,
                            entity: Entity,
                            context: list[Event],
                            choices: list[Event],
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
            a question contains context, choices and target.
        """
        doc_id = doc.doc_id
        entities = doc.entities
        context = [_.quadruple(entity, entities) for _ in context]
        choices = [_.quadruple(entity, entities) for _ in choices]
        return context, choices, target

    def indexing_a_question(self, question):
        """Indexing a question.

        Args:
            question: the elements needed for question.

        Returns:
            an indexed question contains context, choices and target.
        """
        context, choices, target = question
        word_dict = self._word_dict
        role_dict = self._role_dict
        context_idx = [_indexing_event(_, word_dict, role_dict) for _ in context]
        choices_idx = [_indexing_event(_, word_dict, role_dict) for _ in choices]
        return context_idx, choices_idx, target
