
class DataBase:
    # to be figured out
    pass

class EvalEngine():

    def __init__(self, yaml_dict):
        # look into eval_engines/spectre/script_test/cs_meas_man.py for an example
        pass

    @classmethod
    def from_yaml(self, yaml_file):
        # creates the object from a yaml file
        pass

    def generate_rand_samples(self, n:int):
        """
        Generates n samples randomly, with their evaluation results. If debug is False errors are
        not raised and only valid designs (with no errors) will be returned.
        :param n:
            number of samples to be generated
        :param debug:
            if True evaluations will run in a single thread and any error encountered  along the
            way will raise an exception
        :return:
            List of design objects with their spec values.
        """
        pass

    def evaluate(self, design_list: List[Design], debug: bool = False) -> List[Dict[str, float]]:
        """
        Evaluates designs in design_list
        :param design_list:
            List of designs
        :param debug:
            if True exceptions are raised if encountered, otherwise they are ignored.
            Simulations run int series when debug is True.
        :return:
            a list of d
        """

        results = None
        pass
        return results

class ComputationalGraph:

    def __init__(self, fname):
        pass


    def characterize_design(self, *args, **kwargs):
        pass


    def get_results(self, *args, **kwargs):
        pass


class Design(list):

    def __init__(self, spec_range, id_encoder, seq=()):
        """
        Design class holds the current list of parameters to be execute for this specific design
        instance
        :param spec_range: Dict[Str : [a,b]] -> kwrds are used for spec property creation of
        Design objects
        :param seq: input parameter vector as a List
        """
        # properties:
        # cost
        # specs
        # hash
        pass




