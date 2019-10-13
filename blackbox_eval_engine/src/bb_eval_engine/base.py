from typing import Optional, Dict, Any, Sequence

import abc
import yaml

from .util.design import Design


class EvaluationEngineBase(abc.ABC):

    def __init__(self, yaml_fname: Optional[str] = None,
                 specs: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        if yaml_fname:
            with open(yaml_fname, 'r') as f:
                specs = yaml.load(f, Loader=yaml.FullLoader)

        self.specs = specs

    @abc.abstractmethod
    def get_rand_sample(self) -> Design:
        """
        implement this method to implement the random generation and meaning of each design
        Returns
        -------
        design: Design
            design object.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def interpret(self, design: Design) -> Dict[str, Any]:
        """
        implement this method to change the interpretation of each design
        Parameters
        ----------
        design: Design
            design object under consideration
        Returns
        -------
        values: Dict[str, Any]
            a dictionary representing the values of design parameters
        """
        raise NotImplementedError

    @abc.abstractmethod
    def generate_rand_designs(self, n: int = 1, evaluate: bool = False,
                              *args, **kwargs) -> Sequence[Design]:
        """
        Generates a random database of Design elements.

        Parameters
        ----------
        n: int
            number of individuals in the data base.
        evaluate: bool
            True to evaluate the value of each design and populate its attributes.

        Returns
        -------
        database: Sequence[Design]
            a sequence of design objects
        """
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, designs: Sequence[Design], *args, **kwargs) -> Any:
        """
        Evaluates (runs simulations) a sequence of design objects.
        Parameters
        ----------
        designs: Sequence[Design]
            input designs to be evaluated
        Returns
        -------
            Anything
        """
        raise NotImplementedError
