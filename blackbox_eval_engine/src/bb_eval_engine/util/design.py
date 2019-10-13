from typing import Optional, Sequence, Any, Dict
from copy import deepcopy, copy

from .immutable import ImmutableType, to_immutable


class Design:
    def __init__(self, value: Sequence[int], attrs: Optional[Sequence[str]] = None):

        self._value = value
        self._attrs: Dict[str, Any] = {'value': value}
        if attrs is not None:
            for k in attrs:
                self._attrs[k] = None

        self._value_hashable: ImmutableType = to_immutable(self._value)

    @property
    def value(self):
        return self._value_hashable

    @property
    def specs(self):
        return self._attrs

    def id(self, id_encoder):
        return id_encoder.convert_list_2_id(self._value)

    def copy(self):
        new_dsn = Design(copy(self._value))
        for k in self._attrs:
            new_dsn[k] = deepcopy(self._attrs[k])
        return new_dsn

    def __getattr__(self, item):
        return self._attrs[item]

    def __getitem__(self, item):
        return self._attrs[item]

    def __setitem__(self, key: str, value: Any = None):
        self._attrs[key] = value

    def __str__(self):
        return self._value.__str__()

    def __repr__(self):
        return self._value.__repr__()

    def __hash__(self):
        if self._value_hashable is None:
            raise ValueError('attribute value hashable is not set')

        return hash(self._value_hashable)

    def __eq__(self, other):
        if self.value is None or other.value is None:
            raise ValueError('attribute value hashable is not set')
        return hash(self) == hash(other)

    # handle pickling, because of the __getattr__ implementation it will look for __getstate__ in
    # self._attrs and raises KeyError
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    # TODO: Clean this mess, make it agnostic to the type of design (i.e circuits)
    @property
    def cost(self):
        try:
            return self._attrs['cost']
        except KeyError:
            raise KeyError('cost is not an attribute of design')

    def set_parents_and_sibling(self, parent1, parent2, sibling):
        self['parent1'] = parent1
        self['parent2'] = parent2
        self['sibiling'] = sibling

    def is_init_population(self):
        if self['parent1'] is None:
            return True
        else:
            return False

    def is_mutated(self):
        if self['parent1'] is not None:
            if self['parent2'] is None:
                return True
        else:
            return False

    @staticmethod
    def genocide(*args):
        for dsn in args:
            dsn.parent1 = None
            dsn.parent2 = None
            dsn.sibling = None
