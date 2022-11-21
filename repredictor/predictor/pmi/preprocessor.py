import logging
import os
import pickle
from typing import List

from tqdm import tqdm

from repredictor.predictor.base.basic_preprocessor import BasicPreprocessor
from repredictor.utils.document import Document
from repredictor.utils.entity import Entity
from repredictor.utils.event import Event


class PmiPreprocessor(BasicPreprocessor):
    """PMI preprocessor."""

    def __init__(self, config: dict):
        super(PmiPreprocessor, self).__init__(config)
        self._logger = logging.getLogger("repredictor.PMIPredictor.preprocessor")

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

        To maintain consistency of input, we pass many arguments,
        though only a few is needed here.

        Args:
            train_dir (str): train documents path.
            question_dir (str): questions directory.
            neg_pool_path (str): not used.
            context_size (int, optional): not used.
            num_distractors (int, optional): not used.
            slice_size (int, optional): not used.
            num_questions (int, optional): not used.
            overwrite (bool, optional): whether to overwrite old files.
        """
        if os.path.exists(question_dir) and not overwrite:
            self._logger.info(f"{question_dir} already exists.")
        else:
            # Load stoplist
            if self._stoplist is None:
                self.load_stoplist()
            stoplist = self._stoplist
            # For each document, generate all its narrative chains.
            self._logger.info(f"Generating question set from {train_dir} ...")
            questions = []
            fn_list = os.listdir(train_dir)
            with tqdm(total=len(fn_list), ascii=True) as pbar:
                pbar.set_description("Processed documents")
                for fn in fn_list:
                    fp = os.path.join(train_dir, fn)
                    doc = Document.from_file(fp)
                    for entity, chain in doc.get_chains(stoplist):
                        # Filter short chains
                        if len(chain) <= context_size:
                            continue
                        chain = [e.predicate_gr(entity) for e in chain]
                        questions.append(chain)
                    pbar.update(1)
            with open(question_dir, "wb") as f:
                pickle.dump(questions, f)
            self._logger.info(f"{question_dir} generated, "
                              f"totally {len(questions)} questions.")

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
        context = [event.predicate_gr(entity) for event in context]
        choices = [event.predicate_gr(entity) for event in choices]
        return context, choices, target

    def indexing_a_question(self, question):
        raise NotImplemented

    def preprocess(self):
        """Preprocess."""
        preprocess_dir = self._preprocess_dir
        os.makedirs(preprocess_dir, exist_ok=True)
        train_doc_dir = os.path.join(self._data_dir, "rich_docs", "train")
        dev_questions_dir = os.path.join(self._data_dir, "eval", "dev")
        test_questions_dir = os.path.join(self._data_dir, "eval", "test")
        train_neg_pool_path = os.path.join(preprocess_dir, "neg_pool.pkl")
        overwrite = self._overwrite
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
        # Generate train question set
        train_question_path = os.path.join(preprocess_dir, "train.pkl")
        self.generate_train_set(
            train_dir=train_doc_dir,
            question_dir=train_question_path,
            neg_pool_path=train_neg_pool_path,
            num_questions=None,
            overwrite=overwrite)
