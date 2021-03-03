import pickle

import numpy as np
import pandas as pd
import torch

class HierDataLoader(object):
    """
    default: load the hatt solutions
    ag-0: load the aggregation solutions
    """
    def __init__(self, workdir, pattern = "default", device = "cuda:0"):
        super(HierDataLoader, self).__init__()
        self._pattern = pattern
        self._init_vec = pickle.load(open(str(workdir+'/data/initial_vectors/init_vec').replace("\\", "/"), 'rb'))
        self.device = device

        if self._pattern == "default":
            self.relation_levels_Tensor = torch.LongTensor(self._init_vec['relation_levels'])
            self.relation_levels_pd = pd.DataFrame(self._init_vec['relation_levels'], columns=['p_index', 'index'])
            self.relation_levels_np = self.relation_levels_pd.to_numpy()
            self.relation_level_layer = (1 + np.max(self._init_vec['relation_levels'], 0)).astype(np.int32)
        else:
            raise Exception("No patterns")

    @property
    def pattern(self):
        return self._pattern