"""Basic preprocessor."""

import os
import pickle
import random
from abc import ABC, abstractmethod

from collections import Counter
from copy import deepcopy

from tqdm import tqdm

from repredictor.utils.document import Document

from repredictor.utils.entity import Entity
from repredictor.utils.event import Event


__all__ = ["BasicPreprocessor"]


def _sample_chain_from_doc(doc: Document,
                           min_len: int = None,
                           stoplist: list[tuple[str, str]] = None
                           ) -> tuple[Entity or None, list[Event]]:
    """Sample an event chain from the document.

    Args:
        doc (Document): the document.
        min_len (int, optional): the min length of the chain.
            Defaults to None.
        stoplist (list[tuple[str, str]], optional): event stop list.
            Defaults to None.

    Returns:
        tuple[Entity or None, list[Event]]: the sampled protagonist and chain.
    """
    min_len = min_len or 0
    chains = [(entity, chain) for entity, chain in doc.get_chains(stoplist)
              if len(chain) > min_len]
    if len(chains) == 0:
        return None, []
    else:
        return random.choice(chains)


def _generate_distractor_event(neg_pool: list[Event],
                               entity: Entity,
                               non_protagonist_entities: list[Entity]
                               ) -> Event:
    """Generate a distractor event from a random negative event.

    Args:
        neg_pool (list[Event]): the negative event pool.
        entity (Entity): the protagonist.
        non_protagonist_entities (list[Entity]):
            other entities in the document.

    Returns:
        Event: the generated distractor event.
    """
    # Randomly select an event from negative event pool
    neg_protagonist, neg_event = random.choice(neg_pool)
    event = deepcopy(neg_event)
    protagonist_id = neg_protagonist.ent_id
    # Replace each entity argument with a non-protagonist entity
    for role in event.roles:
        if role.ent_id is not None:
            if role.ent_id == protagonist_id:
                # Replace the argument with protagonist
                role.ent_id = entity.ent_id
                role.value = entity.head
                role.concept = entity.concept
            elif len(non_protagonist_entities) > 0:
                # In some cases, there are no other entities
                rand_ent = random.choice(non_protagonist_entities)
                role.ent_id = rand_ent.ent_id
                role.value = rand_ent.head
                role.concept = rand_ent.concept
            else:
                # if there are no other entities, view the arguments as common words
                role.ent_id = None
    return event


class BasicPreprocessor(ABC):
    """Basic preprocessor.

    The abstract methods must be implemented:
        - ``generate_a_question``
        - ``indexing_a_question``
    """

    def __init__(self, config: dict):
        """Construction method for BasicPreprocessor class.

        Args:
            config (dict): configuration.
        """
        self._config = config
        # Source data directory
        self._data_dir = config["data_dir"]
        # Work directory
        self._work_dir = config["work_dir"]
        # Whether to use progress_bar in this model
        self._progress_bar = config["progress_bar"]
        # Whether to overwrite existing files
        self._overwrite = config["overwrite"]
        # Model type: In most cases, same model type shares same preprocessor
        self._model_type = config["model"]["type"]
        # Preprocess data dir
        self._preprocess_dir = os.path.join(
            self._work_dir, f"{self._model_type}_data")
        # Logger
        self._logger = None
        # ===== Common Hyper-parameters =====
        self._min_count = config["preprocess"]["min_freq"]
        self._seed = config["preprocess"]["seed"]
        self._seq_len = config["model"]["seq_len"]
        self._word_dict = None
        self._role_dict = None
        self._concept_dict = None
        self._stoplist = None

    @property
    def preprocess_dir(self):
        """Get preprocess directory."""
        return self._preprocess_dir

    def load_stoplist(self, fp: str = None) -> None:
        """Load stop verb list.

        Args:
            fp (str, optional): file path.
                Defaults to None.
        """
        if fp is None:
            fp = "data/stoplist.txt"
        stoplist = []
        with open(fp, "r") as f:
            for line in f:
                stoplist.append(tuple(line.strip().split("\t")))
        self._stoplist = stoplist

    def load_word_dict(self, fp: str = None) -> None:
        """Load word dictionary.

        Args:
            fp (str, optional): file path.
                Defaults to None.
        """
        if fp is None:
            word_dict_path = os.path.join(
                self._preprocess_dir, "word_dict.pkl")
        else:
            word_dict_path = fp
        with open(word_dict_path, "rb") as f:
            word_dict = pickle.load(f)
        self._word_dict = word_dict

    def load_role_dict(self, fp: str = None) -> None:
        """Load role dictionary.

        Args:
            fp (str, optional): file path.
                Defaults to None.
        """
        if fp is None:
            role_dict_path = os.path.join(
                self._preprocess_dir, "role_dict.pkl")
        else:
            role_dict_path = fp
        with open(role_dict_path, "rb") as f:
            role_dict = pickle.load(f)
        self._role_dict = role_dict

    def load_concept_dict(self, fp: str = None) -> None:
        """Load concept dictionary.

        Args:
            fp (str, optional): file path.
                Defaults to None.
        """
        if fp is None:
            concept_dict_path = os.path.join(
                self._preprocess_dir, "concept_dict.pkl")
        else:
            concept_dict_path = fp
        with open(concept_dict_path, "rb") as f:
            concept_dict = pickle.load(f)
        self._concept_dict = concept_dict

    def generate_dictionaries(self,
                              train_dir: str,
                              min_count: int = 10,
                              overwrite: bool = False) -> None:
        """Generate word/role/type dictionary.

        Args:
            train_dir (str): train documents directory.
            min_count (int, optional): min count for words.
                Defaults to 10.
            overwrite (bool, optional): whether to overwrite old files.
                Defaults to False.
        """
        preprocess_dir = self._preprocess_dir
        word_dict_path = os.path.join(preprocess_dir, "word_dict.pkl")
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
            role_counter = Counter()
            concept_counter = Counter()
            self._logger.info(f"Generating dictionaries for {train_dir} ...")
            pbar = None
            if self._progress_bar:
                pbar = tqdm(total=len(fn_list))
                pbar.set_description("Processed documents")
            for fn in fn_list:
                fp = os.path.join(train_dir, fn)
                doc = Document.from_file(fp)
                for entity, chain in doc.get_chains(stoplist):
                    for event in chain:
                        words = event.words(doc.entities)
                        roles = map(lambda x: x.role, event.roles)
                        concepts = map(lambda x: x.concept, event.roles)
                        word_counter.update(words)
                        role_counter.update(roles)
                        concept_counter.update(concepts)
                if pbar is not None:
                    pbar.update(1)
            if pbar is not None:
                pbar.close()
            word_stat = word_counter.most_common()
            word_stat = [(w, f) for w, f in word_stat if f >= min_count]
            role_stat = role_counter.most_common()
            concept_stat = concept_counter.most_common()
            concept_stat = [(w, f) for w, f in concept_stat if f >= min_count]
            word_dict = {"None": 0}
            for w, f in word_stat:
                word_dict.setdefault(w, len(word_dict))
            role_dict = {"None": 0}
            for w, f in role_stat:
                role_dict.setdefault(w, len(role_dict))
            concept_dict = {"None": 0}
            for w, f in concept_stat:
                concept_dict.setdefault(w, len(concept_dict))
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

    def generate_neg_pool(self,
                          dataset_dir: str,
                          neg_pool_path: str,
                          num_events: int = 200000,
                          overwrite: bool = False) -> None:
        """Generate negative event pool for train set.

        Args:
            dataset_dir (str): the dataset directory.
            neg_pool_path (str): the negative event pool path.
            num_events (int): number of negative events to be sampled.
                Defaults to 200000.
            overwrite (bool): whether to overwrite old file.
        """
        if os.path.exists(neg_pool_path) and not overwrite:
            self._logger.info(f"{neg_pool_path} already exists")
        else:
            if self._stoplist is None:
                self.load_stoplist()
            stoplist = self._stoplist
            neg_pool = []
            fn_list = os.listdir(dataset_dir)
            self._logger.info(f"Generating negative events for {dataset_dir} ...")
            with tqdm(total=num_events) as pbar:
                pbar.set_description("Sampling events")
                while len(neg_pool) < num_events:
                    # For each time, we sample a document,
                    # and then sample a chain from the document,
                    # finally we sample (entity, event) tuple from the chain.
                    fn = random.choice(fn_list)
                    fp = os.path.join(dataset_dir, fn)
                    doc = Document.from_file(fp)
                    entity, chain = _sample_chain_from_doc(doc=doc, stoplist=stoplist)
                    if entity is None:
                        # Document with no chains
                        continue
                    event = random.choice(chain)
                    neg_pool.append((entity, event))
                    pbar.update(1)
            with open(neg_pool_path, "wb") as f:
                pickle.dump(neg_pool, f)
            self._logger.info(f"Generate {neg_pool_path}, "
                              f"totally {len(neg_pool)} events.")

    @abstractmethod
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
            the formatted question.
        """

    def generate_eval_set(self,
                          eval_dir: str,
                          question_path: str,
                          num_questions: int = None,
                          overwrite: bool = False) -> None:
        """Generate evaluation question set.

        Args:
            eval_dir (str): the evaluation documents directory.
            question_path (str): the question dataset path.
            num_questions (int, optional): the questions to be generated.
                Defaults to None.
            overwrite (bool, optional): whether to overwrite old file.
                Defaults to False.
        """
        if os.path.exists(question_path) and not overwrite:
            self._logger.info(f"{question_path} already exists.")
        else:
            questions = []
            fn_list = os.listdir(eval_dir)
            num_questions = num_questions or len(fn_list)
            fn_list = fn_list[:num_questions]
            self._logger.info(f"Generating question set from {eval_dir} ...")
            with tqdm(total=len(fn_list)) as pbar:
                pbar.set_description("Processed questions")
                for fn in fn_list:
                    fp = os.path.join(eval_dir, fn)
                    doc = Document.from_file(fp)
                    entity = doc.protagonist
                    context = doc.context
                    choices = doc.choices
                    target = doc.target
                    question = self.generate_a_question(
                        entity=entity,
                        context=context,
                        choices=choices,
                        target=target,
                        doc=doc)
                    questions.append(question)
                    pbar.update(1)
            with open(question_path, "wb") as f:
                pickle.dump(questions, f)
            self._logger.info(f"{question_path} generated, "
                              f"totally {len(questions)} questions.")

    def generate_train_set(self,
                           train_dir: str,
                           question_dir: str,
                           neg_pool_path: str,
                           context_size: int = 8,
                           num_distractors: int = 4,
                           slice_size: int = 1000000,
                           num_questions: int = None,
                           overwrite: bool = False) -> None:
        """Generate train question set.

        Args:
            train_dir (str): train documents directory.
            question_dir (str): questions directory.
            neg_pool_path (str): negative event pool path.
            context_size (int, optional): number of context events.
                Defaults to 8.
            num_distractors (int, optional):
                number of distractor event for each question.
                Defaults to 4.
            slice_size (int, optional): number of questions for each file.
                Defaults to 1000000.
            num_questions (int, optional): number of train samples to be generated.
                Defaults to None.
            overwrite (bool, optional): whether to overwrite old files.
                Defaults to False.
        """
        if len(os.listdir(question_dir)) > 0 and not overwrite:
            self._logger.info(f"{question_dir} already exists.")
        else:
            # Load stoplist
            if self._stoplist is None:
                self.load_stoplist()
            stoplist = self._stoplist
            # Load neg pool
            num_slices = 0
            with open(neg_pool_path, "rb") as f:
                neg_pool = pickle.load(f)
            # For each document, generate all its narrative chains.
            self._logger.info(f"Generating question set from {train_dir} ...")
            questions = []
            fn_list = os.listdir(train_dir)
            total_questions = 0
            with tqdm(total=len(fn_list)) as pbar:
                pbar.set_description("Processed documents")
                for fn in fn_list:
                    if num_questions is not None and total_questions >= num_questions:
                        continue
                    fp = os.path.join(train_dir, fn)
                    doc = Document.from_file(fp)
                    for entity, chain in doc.get_chains(stoplist):
                        # Filter short chains
                        if len(chain) <= context_size:
                            continue
                        # Generate questions
                        non_protagonist_entities = [e for e in doc.entities if e is not entity]
                        for answer_idx in range(context_size, len(chain)):
                            answer = chain[answer_idx]
                            context = chain[answer_idx - context_size:answer_idx]
                            # Generate distractor events
                            distractors = [
                                _generate_distractor_event(neg_pool, entity, non_protagonist_entities)
                                for _ in range(num_distractors)]
                            choices = distractors + [answer]
                            random.shuffle(choices)
                            target = choices.index(answer)
                            question = self.generate_a_question(
                                entity=entity,
                                context=context,
                                choices=choices,
                                target=target,
                                doc=doc)
                            questions.append(question)
                            total_questions += 1
                            # Current slice is full, start a new slice
                            if len(questions) == slice_size:
                                slice_fn = f"train{str(num_slices).zfill(3)}.pkl"
                                slice_fp = os.path.join(question_dir, slice_fn)
                                with open(slice_fp, "wb") as f:
                                    pickle.dump(questions, f)
                                questions = []
                                num_slices += 1
                    pbar.update(1)
            if len(questions) > 0:
                slice_fn = f"train{str(num_slices).zfill(3)}.pkl"
                slice_fp = os.path.join(question_dir, slice_fn)
                with open(slice_fp, "wb") as f:
                    pickle.dump(questions, f)
            self._logger.info(f"{question_dir} generated, "
                              f"totally {total_questions} questions.")

    @abstractmethod
    def indexing_a_question(self, question):
        """Indexing a question.

        Args:
            question: the elements needed for question.

        Returns:
            the indexed question.
        """

    def indexing_questions(self,
                           question_path: str,
                           question_idx_path: str,
                           overwrite=False) -> None:
        """Indexing question dataset.

        Args:
            question_path (str): the question path.
            question_idx_path (str): the indexed question path.
            overwrite (bool, optional): whether to overwrite old files.
                Defaults to None.
        """
        if os.path.exists(question_idx_path) and not overwrite:
            self._logger.info(f"{question_idx_path} already exists.")
        else:
            if self._word_dict is None:
                self.load_word_dict()
            if self._role_dict is None:
                self.load_role_dict()
            if self._concept_dict is None:
                self.load_concept_dict()
            with open(question_path, "rb") as f:
                questions = pickle.load(f)
            questions_idx = [self.indexing_a_question(q) for q in questions]
            with open(question_idx_path, "wb") as f:
                pickle.dump(questions_idx, f)
            self._logger.info(f"Indexing {question_path} to {question_idx_path}.")

    def dataset_statistics(self) -> None:
        """Basic statistics on datasets."""
        preprocess_dir = self._preprocess_dir
        # Count words
        if self._word_dict is None:
            self.load_word_dict()
        self._logger.info(f"Totally {len(self._word_dict)} words in word_dict.")
        # Count roles
        if self._role_dict is None:
            self.load_role_dict()
        self._logger.info(f"Totally {len(self._role_dict)} words in role dict.")
        # Count train question indices
        train_idx_dir = os.path.join(preprocess_dir, "train_idx")
        num_train_questions = 0
        for fn in os.listdir(train_idx_dir):
            train_idx_path = os.path.join(train_idx_dir, fn)
            with open(train_idx_path, "rb") as f:
                train_idx = pickle.load(f)
            num_train_questions += len(train_idx)
        self._logger.info(f"Totally {num_train_questions} questions in train set.")
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

    def preprocess(self):
        """General preprocess steps.

        Basically it executes following steps sequentially:
            - generate_dictionaries
            - generate_eval_set
            - generate_neg_pool
            - generate_train_set
            - indexing_questions
            - dataset_statistics
        """
        preprocess_dir = self._preprocess_dir
        os.makedirs(preprocess_dir, exist_ok=True)
        train_doc_dir = os.path.join(self._data_dir, "rich_docs", "train")
        dev_questions_dir = os.path.join(self._data_dir, "eval", "dev")
        test_questions_dir = os.path.join(self._data_dir, "eval", "test")
        train_neg_pool_path = os.path.join(preprocess_dir, "neg_pool.pkl")
        overwrite = self._overwrite
        # Generate word dictionary
        self.generate_dictionaries(
            train_dir=train_doc_dir,
            min_count=self._min_count)
        # Set random seed
        random.seed(self._seed)
        # Generate evaluation question set
        dev_path = os.path.join(preprocess_dir, "dev.pkl")
        self.generate_eval_set(
            eval_dir=dev_questions_dir,
            question_path=dev_path,
            overwrite=overwrite)
        test_path = os.path.join(preprocess_dir, "test.pkl")
        self.generate_eval_set(
            eval_dir=test_questions_dir,
            question_path=test_path,
            overwrite=overwrite)
        # Generate negative event pool
        num_neg_events = len(os.listdir(train_doc_dir)) * 10
        self.generate_neg_pool(
            dataset_dir=train_doc_dir,
            neg_pool_path=train_neg_pool_path,
            num_events=num_neg_events,
            overwrite=overwrite)
        # Generate train question set
        train_question_dir = os.path.join(preprocess_dir, "train")
        os.makedirs(train_question_dir, exist_ok=True)
        self.generate_train_set(
            train_dir=train_doc_dir,
            question_dir=train_question_dir,
            neg_pool_path=train_neg_pool_path,
            num_questions=None,
            overwrite=overwrite)
        # Indexing questions
        dev_idx_path = os.path.join(preprocess_dir, "dev_idx.pkl")
        self.indexing_questions(
            question_path=dev_path,
            question_idx_path=dev_idx_path,
            overwrite=overwrite)
        test_idx_path = os.path.join(preprocess_dir, "test_idx.pkl")
        self.indexing_questions(
            question_path=test_path,
            question_idx_path=test_idx_path,
            overwrite=overwrite)
        train_idx_dir = os.path.join(preprocess_dir, "train_idx")
        os.makedirs(train_idx_dir, exist_ok=True)
        for fn in os.listdir(train_question_dir):
            train_question_path = os.path.join(train_question_dir, fn)
            train_idx_path = os.path.join(train_idx_dir, fn)
            self.indexing_questions(
                question_path=train_question_path,
                question_idx_path=train_idx_path,
                overwrite=overwrite)
        self.dataset_statistics()
