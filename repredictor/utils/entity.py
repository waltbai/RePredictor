"""Entity class."""

# Pronouns
from collections import Counter


__all__ = ["Entity", "get_headword_for_mention"]


PRONOUNS = [
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "myself", "yourself", "himself", "herself", "itself", "ourself", "ourselves", "themselves",
    "my", "your", "his", "its", "it's", "our", "their",
    "mine", "yours", "ours", "theirs",
    "this", "that", "those", "these",
    "-", ",",
]
# Load stop word list
with open("data/english_stopwords.txt", "r") as f:
    STOPWORDS = f.read().splitlines()


def filter_words_in_mention(words: list[str]):
    """Filter stop words and pronouns in mention.

    Args:
        words (list[str]): the word list.

    Returns:
        list[str]: the filtered word list.
    """
    return [w for w in words if w not in PRONOUNS and w not in STOPWORDS]


def get_headword_for_mention(mention: list[str]):
    """Get headword for mention.

    Args:
        mention (list[str]): the mention word list

    Returns:
        str: the headword
    """
    mention = [w.lower() for w in mention]
    # Filter stop words
    words = filter_words_in_mention(mention)
    # Filter 1-letter word
    words = [w for w in words if len(w) > 1]
    # Use the rightmost word as the mention head.
    if len(words) > 0:
        return words[-1]
    else:
        return "None"


class Entity:
    """Entity class."""

    def __init__(self,
                 mentions: list[list[str]],
                 ent_id: int,
                 head: str = None,
                 salient_mention: list[str] = None,
                 concept: str = None):
        """Construction method for Entity.

        Args:
            mentions (list[list[str]]): the co-reference chain.
            ent_id (int): the entity ID.
            head (str, optional): the headword for this entity.
                Defaults to None.
            salient_mention (list[str], optional): the salient mention for this entity
                Defaults to None.
            concept (str, optional): the entity type for this entity.
                Defaults to None.
        """
        self.mentions = mentions
        self.ent_id = ent_id
        self.head = head or self.get_head()
        self.salient_mention = salient_mention or self.get_salient_mention()
        self.concept = concept

    def __repr__(self):
        return f"[{self.ent_id}, {self.head}, {self.salient_mention}, {self.concept}]"

    def get_head(self):
        """Get mention head.

        Similar with G&C16.

        Returns:
            str: entity headword.
        """
        entity_head_words = Counter()
        for mention in self.mentions:
            headword = get_headword_for_mention(mention)
            entity_head_words.update([headword])
        if len(entity_head_words) > 0:
            return entity_head_words.most_common()[0][0]
        else:
            return "None"

    def get_salient_mention(self):
        """Get salient mention.

        Returns:
            list[str]: the most salient mention.
        """
        salient_mention = []
        for mention in self.mentions:
            # Filter stopwords and pronouns.
            mention = [w.lower() for w in mention]
            words = filter_words_in_mention(mention)
            words = [w for w in words if len(w) > 1]
            # Use the longest mention as the salient mention
            if len(words) > len(salient_mention):
                salient_mention = words
        return salient_mention
