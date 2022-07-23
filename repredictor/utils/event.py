"""Event class."""
from repredictor.utils.entity import Entity, get_headword_for_mention


class Role:
    """Role class."""

    def __init__(self,
                 role: str,
                 value: list[str],
                 concept: str,
                 ent_id: int):
        """Construction method for Role.

        Args:
            role (str): the role type.
            value (list[str]): the argument word list.
            concept (str): the entity type of the argument.
            ent_id (int): the entity ID of the argument.
        """
        self.role = role
        self.value = value
        self.concept = concept
        self.ent_id = ent_id

    def __repr__(self) -> str:
        return f"-[{self.role}]->({self.value}, {self.concept}, {self.ent_id})"

    def __eq__(self, other) -> bool:
        return self.role == other.role \
            and self.value == other.value \
            and self.concept == other.concept \
            and self.ent_id == other.ent_id

    def to_json(self) -> dict:
        """To json object."""
        return {
            "role": self.role,
            "value": self.value,
            "concept": self.concept,
            "ent_id": self.ent_id,
        }


def find_arg_word(role: Role,
                  entities: list[Entity]) -> str:
    """Find the headword of the argument.

    Args:
        role (Role): the role.
        entities (list[Entity]): the entity list.

    Returns:
        str: the headword of the argument.
    """
    if role.ent_id is not None:
        assert role.ent_id < len(entities), f"\n{role}\n {entities}"
        value = entities[role.ent_id].head
    else:
        value = get_headword_for_mention(role.value)
    return value


def find_arg(roles: list[Role],
             relation: str,
             entities: list[Entity]) -> str:
    """Find argument according to its role type.

    Args:
        roles (list[Role]): the role list.
        relation (str): the target role type.
        entities (list[Entity]): the entity list.

    Returns:
        str: the headword of the argument.
    """
    value = "None"
    for r in roles:
        if r.role == relation:
            value = find_arg_word(r, entities)
            break
    return value


def find_concept(role: Role, entities: list[Entity]) -> str:
    """Find corresponding concept.

    Args:
        role (Role): the target role.
        entities (list[Entity]): the entity list.

    Returns:
        str: the event type
    """
    if role.ent_id is not None:
        return entities[role.ent_id].concept
    else:
        return role.concept


class Event:
    """Event class."""

    def __init__(self,
                 pb_frame: str,
                 verb_pos: int,
                 sent_id: int,
                 roles: list[dict]):
        """Construction method for Event.

        Args:
            pb_frame (str): the propbank frame for the verb.
            verb_pos (int): the index of verb in the sentence.
            sent_id (int): the sentence ID.
            roles (list[dict]): the argument list.
        """
        self.pb_frame = pb_frame    # propbank frame
        self.verb_pos = verb_pos
        self.sent_id = sent_id
        self.roles = [Role(**r) for r in roles]

    def __repr__(self) -> str:
        s = f"{self.pb_frame}:\n"
        for r in self.roles:
            s = s + f"\t{r}\n"
        return s

    def __eq__(self, other) -> bool:
        result = self.pb_frame == other.pb_frame \
            and self.verb_pos == other.verb_pos \
            and self.sent_id == other.sent_id \
            and self.roles == other.roles
        return result

    @property
    def position(self) -> tuple[int, int]:
        """Tuple position representation.

        Returns:
            int: sentence id
            int: verb position
        """
        return self.sent_id, self.verb_pos

    def to_json(self) -> dict:
        """To json object."""
        return {
            "pb_frame": self.pb_frame,
            "verb_pos": self.verb_pos,
            "sent_id": self.sent_id,
            "roles": [r.to_json() for r in self.roles]
        }

    def contains(self, entity: Entity) -> bool:
        """If one of the event argument is the entity.

        Args:
            entity (Entity): the entity.

        Returns:
            bool: if the entity participates in this event.
        """
        for r in self.roles:
            if r.ent_id == entity.ent_id:
                return True
        return False

    def find_role(self, entity: Entity) -> str or None:
        """Find role for an entity.

        Args:
            entity (Entity): the entity.

        Returns:
            str or None: role type of the entity.
        """
        for r in self.roles:
            if r.ent_id == entity.ent_id:
                return r.role
        return None

    def predicate_gr(self, entity: Entity) -> tuple[str, str]:
        """Return (verb, role) tuple.

        Args:
            entity (Entity): the protagonist.

        Returns:
            tuple[str, str]: the (verb, role) tuple.
        """
        return self.pb_frame, self.find_role(entity)

    def quintuple(self,
                  protagonist: Entity,
                  entities: list[Entity]) -> tuple[str, str, str, str, str]:
        """Return (v, a0, a1, a2, role) quintuple.

        Args:
            protagonist (Entity): the protagonist.
            entities (list[Entity]): the entity list.

        Returns:
            [str, str, str, str, str]: the quintuple representation.
        """
        verb = self.pb_frame
        role = self.find_role(protagonist)
        # protagonist_id = protagonist.ent_id
        # protagonist = protagonist.head
        arg0 = find_arg(self.roles, ":ARG0", entities)
        arg1 = find_arg(self.roles, ":ARG1", entities)
        arg2 = find_arg(self.roles, ":ARG2", entities)
        return verb, arg0, arg1, arg2, role

    def quadruple(self,
                  protagonist: Entity,
                  entities: list[Entity]) -> tuple[str, str, str, str]:
        """Return (predicate_gr, a0, a1, a2) quadruple.

        Args:
            protagonist (Entity): the protagonist.
            entities (list[Entity]): the entity list.

        Returns:
            tuple[str, str, str, str]: the quadruple representation.
        """
        verb, arg0, arg1, arg2, role = self.quintuple(protagonist, entities)
        predicate_gr = f"{verb}:{role}"
        return predicate_gr, arg0, arg1, arg2

    def rich_repr(self,
                  protagonist: Entity,
                  entities: list[Entity]
                  ) -> tuple[str, str or None, list[str], list[str], list[str]]:
        """Return rich event representation, as role-value list.

        Args:
            protagonist (Entity): the protagonist.
            entities (list[Entity]): the entity list.

        Returns:
            tuple[str, str or None, list[str], list[str], list[str]]:
                the event type, the protagonist role and the arguments.
        """
        verb = self.pb_frame
        role = self.find_role(protagonist)
        roles = [r.role for r in self.roles]
        values = [find_arg_word(r, entities) for r in self.roles]
        # Add concepts for each role
        concepts = [find_concept(r, entities) for r in self.roles]
        return verb, role, roles, values, concepts

    def entities(self) -> list[int]:
        """Return ids of all entities participate in this event.

        Returns:
            list[int]: the entity IDs.
        """
        ents = set()
        for r in self.roles:
            if r.ent_id is not None:
                ents.add(r.ent_id)
        return sorted(list(ents))

    def arguments(self, entities) -> list[str]:
        """Return argument head words except None.

        Returns:
            list[str]: the argument headword list.
        """
        values = [find_arg_word(r, entities) for r in self.roles]
        values = [_ for _ in values if _ != "None"]
        return values

    def words(self, entities) -> list[str]:
        """Return all words in event.

        Returns:
            list[str]: the word list.
        """
        result = [self.pb_frame]
        for r in self.roles:
            result.append(find_arg_word(r, entities))
        result = [w for w in result if w != "None"]
        return result
