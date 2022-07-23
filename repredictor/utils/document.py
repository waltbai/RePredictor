"""Document class."""
import json
import os

from repredictor.utils.entity import Entity
from repredictor.utils.event import Event


def _between(pos: tuple[int, int],
             start_pos: tuple[int, int],
             end_pos: tuple[int, int]) -> bool:
    """If the position is between the start and the end.

    Args:
        pos (tuple[int, int]): the verb position.
        start_pos (tuple[int, int]): the start position.
        end_pos (tuple[int, int]): the end position.

    Returns:
        bool: whether pos is between start and end.
    """
    start_check = start_pos is None or start_pos <= pos
    end_check = end_pos is None or end_pos >= pos
    return start_check and end_check


class Document:
    """Document class."""

    def __init__(self,
                 doc_id: str,
                 entities: list[Entity],
                 events: list[Event],
                 protagonist: Entity = None,
                 context: list[Event] = None,
                 choices: list[Event] = None,
                 target: int = None):
        """Construction method for Document.

        Args:
            doc_id (str): the document ID.
            entities (list[Entity]): entity list.
            events (list[Event]): event list.
            protagonist (Entity): the protagonist used for dev/test docs.
                Defaults to None.
            context (list[Event]): the context used for dev/test docs.
                Defaults to None.
            choices (list[Event]): the choices used for dev/test docs.
                Defaults to None.
            target (int): the answer used for dev/test docs.
                Defaults to None
        """
        self.doc_id = doc_id
        self.entities = entities
        self.events = events
        self.context = context
        self.choices = choices
        self.target = target
        self.protagonist = protagonist

    @classmethod
    def from_file(cls, fpath, tokens: list[str] = None):
        """Read document from path.

        Args:
            fpath (str): file path.
            tokens (list[str], optional): tokens of the original text.
                Defaults to None.
        """
        with open(fpath, "r") as f:
            doc = json.load(f)
        tokens = tokens or []
        doc.setdefault("tokens", tokens)
        doc_id = doc["doc_id"]
        entities = [Entity(**e) for e in doc["entities"]]
        events = [Event(**e) for e in doc["events"]]
        context, choices, target, protagonist = None, None, None, None
        if "context" in doc:
            context = [Event(**e) for e in doc["context"]]
        if "choices" in doc:
            choices = [Event(**e) for e in doc["choices"]]
        if "target" in doc:
            target = doc["target"]
        if "entity_id" in doc:
            protagonist = entities[doc["entity_id"]]
        return cls(doc_id=doc_id,
                   entities=entities,
                   events=events,
                   protagonist=protagonist,
                   context=context,
                   choices=choices,
                   target=target)

    def get_chain_by_entity_id(self,
                               entity: Entity,
                               stoplist: list[tuple[str, str]] = None
                               ) -> list[Event]:
        """Get chain by entity id.

        Args:
            entity (Entity): the protagonist.
            stoplist (list[tuple[str, str]], optional): the stop verb list.
                Defaults to None.

        Returns:
            list[Event]: the event list that entity participate in.
        """
        if stoplist is None:
            return [e for e in self.events if e.contains(entity)]
        else:
            return [e for e in self.events
                    if e.contains(entity) and e.predicate_gr(entity) not in stoplist]

    def get_chains(self,
                   stoplist: list[tuple[str, str]] = None):
        """Get entities and chains.

        Args:
            stoplist (list[tuple[str, str]], optional): the stop verb list.
                Defaults to None.

        Yields:
            tuple[Entity, list[Event]]: protagonist and the corresponding event chain.
        """
        for entity in self.entities:
            yield entity, self.get_chain_by_entity_id(entity, stoplist)

    def get_events(self,
                   start_pos: tuple[int, int] = None,
                   end_pos: tuple[int, int] = None) -> list[Event]:
        """Get events between start and end position.

        Args:
            start_pos (tuple[int, int]): start position.
                Defaults to None.
            end_pos (tuple[int, int]): end position.
                Defaults to None.

        Returns:
            list[Event]: the event list between start and end.
        """
        return [
            e for e in self.events
            if _between(e.position, start_pos, end_pos)]


def document_iterator(doc_dir: str):
    """Iterate each document.

    Args:
        doc_dir (str): the documents' directory.

    Yields:
        Document: document.
    """
    for root, dirs, files in os.walk(doc_dir):
        for f in files:
            if f.endswith(".txt"):
                fpath = os.path.join(root, f)
                doc = Document.from_file(fpath=fpath)
                yield doc
