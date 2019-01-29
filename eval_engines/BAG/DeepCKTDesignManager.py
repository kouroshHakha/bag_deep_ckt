from bag.simulation.core import DesignManager
from bag.core import BagProject
from bag.io import read_yaml
from copy import deepcopy

class DeepCKTDesignManager(DesignManager):
    #TODO Come up with a better DeisgnManager
    # 1. It should associate schematic and layout together always
    # 2. Should not replace layout and dump the schematic in somewhere else
    # 3. make gen_wrapper=True if wrapper is given, if not make it false
    def get_layout_params(self, val_list):
        # type: (Tuple[Any, ...]) -> Dict[str, Any]
        """Returns the layout dictionary from the given sweep parameter values. Overwritten to incorporate
        the discrete evaluation problem, instead of a sweep"""

        # sanity check
        assert len(self.swp_var_list) == 1 and 'swp_spec_file' in self.swp_var_list , \
            "when using DeepCKTDesignManager only replacting file name (swp_spec_file) should be part of the " \
            "sweep_params dictionary"

        lay_params = deepcopy(self.specs['layout_params'])
        yaml_fname = self.specs['root_dir']+'/gen_yamls/swp_spec_files/' + val_list[0] + '.yaml'
        print(yaml_fname)
        updated_top_specs = read_yaml(yaml_fname)
        new_lay_params = updated_top_specs['layout_params']
        lay_params.update(new_lay_params)
        return lay_params
