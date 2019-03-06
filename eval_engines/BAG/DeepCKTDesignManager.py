from bag.simulation.core import DesignManager
from bag.core import BagProject
from bag.io import read_yaml
from copy import deepcopy

from bag.layout import RoutingGrid, TemplateDB
from bag import float_to_si_string
from bag.io import read_yaml, open_file, load_sim_results, save_sim_results, load_sim_file
from bag.concurrent.core import batch_async_task
from bag import BagProject

class DeepCKTDesignManager(DesignManager):
    # Modifications:
    #   get_layout -> replaces the lay_params with the correct values
    #   create_designs -> doesn't create the template data base, rather
    #   it assumes it is already set, which happens from outside

    # TODO Come up with a better DeisgnManager
    # 1. It should associate schematic and layout together always
    # 2. Should not replace layout and dump the schematic in somewhere else
    # 3. make gen_wrapper=True if wrapper is given, if not make it false

    def __init__(self, *args, **kwargs):
        self._temp_db = None
        DesignManager.__init__(self, *args, **kwargs)

    def set_tdb(self, tdb):
        self._temp_db = tdb

    def get_layout_params(self, val_list):
        # type: (Tuple[Any, ...]) -> Dict[str, Any]
        """Returns the layout dictionary from the given sweep parameter values. Overwritten to incorporate
        the discrete evaluation problem, instead of a sweep"""

        # sanity check
        assert len(self.swp_var_list) == 1 and 'swp_spec_file' in self.swp_var_list , \
            "when using DeepCKTDesignManager for replacing file name, just (swp_spec_file) should be part of the " \
            "sweep_params dictionary"

        lay_params = deepcopy(self.specs['layout_params'])
        yaml_fname = self.specs['root_dir']+'/gen_yamls/swp_spec_files/' + val_list[0] + '.yaml'
        print(yaml_fname)
        updated_top_specs = read_yaml(yaml_fname)
        new_lay_params = updated_top_specs['layout_params']
        lay_params.update(new_lay_params)
        return lay_params

    def create_designs(self, create_layout):
        # type: (bool) -> None
        """Create DUT schematics/layouts.
        """
        if self.prj is None:
            raise ValueError('BagProject instance is not given.')
        ##########################################################
        ############## Change made here:
        ##########################################################
        if self._temp_db is None:
            self._temp_db = self.make_tdb()
        temp_db = self._temp_db

        # make layouts
        dsn_name_list, lay_params_list, combo_list_list = [], [], []
        for combo_list in self.get_combinations_iter():
            dsn_name = self.get_design_name(combo_list)
            lay_params = self.get_layout_params(combo_list)
            dsn_name_list.append(dsn_name)
            lay_params_list.append(lay_params)
            combo_list_list.append(combo_list)
        import time
        start = time.time()
        if create_layout:
            print('creating all layouts.')
            sch_params_list = self.create_dut_layouts(lay_params_list, dsn_name_list, temp_db)
        else:
            print('schematic simulation, skipping layouts.')
            sch_params_list = [self.get_schematic_params(combo_list)
                               for combo_list in self.get_combinations_iter()]
        print("Layout Creation time:{}".format(time.time()-start))
        start = time.time()
        print('creating all schematics.')
        self.create_dut_schematics(sch_params_list, dsn_name_list, gen_wrappers=True)

        print("Schematic Creation time:{}".format(time.time()-start))
        print('design generation done.')

    def characterize_designs(self, generate=True, measure=True, load_from_file=False):
        # type: (bool, bool, bool) -> None
        """Sweep all designs and characterize them.

        Parameters
        ----------
        generate : bool
            If True, create schematic/layout and run LVS/RCX.
        measure : bool
            If True, run all measurements.
        load_from_file : bool
            If True, measurements will load existing simulation data
            instead of running simulations.
        """
        # import time
        # start = time.time()
        if generate:
            extract = self.specs['view_name'] != 'schematic'
            self.create_designs(extract)
        else:
            extract = False
        # print("design_creation_time = {}".format(time.time()-start))
        rcx_params = self.specs.get('rcx_params', None)
        impl_lib = self.specs['impl_lib']
        dsn_name_list = [self.get_design_name(combo_list)
                         for combo_list in self.get_combinations_iter()]

        coro_list = [self.main_task(impl_lib, dsn_name, rcx_params, extract=extract,
                                    measure=measure, load_from_file=load_from_file)
                     for dsn_name in dsn_name_list]

        results = batch_async_task(coro_list)
        # if results is not None:
        #     for val in results:
        #         if isinstance(val, Exception):
        #             raise val
        return results