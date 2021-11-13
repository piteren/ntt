import math
from typing import Any

from ptools.lipytools.little_methods import prep_folder
from ptools.lipytools.plots import two_dim
from ptools.mpython.omp import OMPRunner, RunningWorker


from model_train import train_model


class EvalTrainer(RunningWorker):

    def __init__(self, devices):
        self.devices = devices

    def process(self, **kwargs) -> Any:
        return train_model(
            devices=        self.devices,
            hpmser_mode=    True,
            **kwargs)


def evaluate_param(
        param: str,
        rng: list,              # param range
        preset_name: str,
        log=            False,  # log / lin scale
        type_int=       False,  # False for float else int
        num_samples=    300,
        devices=        [0,1]*5):

    seed = [1/num_samples * x for x in range(num_samples)]
    if log:
        rng = [math.log10(rng[0]), math.log10(rng[1])]
    params = [rng[0] + (rng[1]-rng[0]) * s for s in seed]
    if log:
        params = [10 ** p for p in params]
    if type_int: params = list(set([int(p) for p in params]))

    tasks = [{
        'preset_name':  preset_name,
        param:          p} for p in params]

    ompr = OMPRunner(
        rw_class=       EvalTrainer,
        rw_lifetime=    1,
        devices=        devices,
        verb=           1)

    results = ompr.process(tasks)
    yx = list(zip(results,params))
    yx.sort(key= lambda x:x[1])

    save_FD = '_param_eval'
    prep_folder(save_FD)
    two_dim(
        y=          yx,
        name=       f'eval_{param}_for_{preset_name}',
        save_FD=    save_FD,
        xlogscale=  log,
        legend_loc= 'lower right')


if __name__ == '__main__':

    #"""
    # iLR
    eval_base = {
        'param':    'iLR',
        #'rng':      [1e-6, 1e-0],
        'rng':      [1e-3, 1e-1],
        #'log':      True
    }
    """
    # hid_width
    eval_base = {
        'param':    'hid_width',
        'rng':      [2, 1200],
        'type_int': True,
    }
    #"""

    for preset in [
        'use_base_U1',
        #'use_one_hidden',
        #'use_hidden_stack'
    ]:

        ed = {}
        ed.update(eval_base)
        ed['preset_name'] = preset

        evaluate_param(**ed)