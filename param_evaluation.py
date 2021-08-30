import math
from typing import Any

from ptools.lipytools.little_methods import prep_folder
from ptools.lipytools.plots import two_dim
from ptools.mpython.omp import OMPRunnerGPU, RunningWorkerGPU


from model_train import train_model


class EvalTrainer(RunningWorkerGPU):

    def run(self, **kwargs) -> Any:
        return train_model(
            devices=        self.devices,
            hpmser_mode=    True,
            **kwargs)


def evaluate_param(
        param: str,
        rng: list,              # param range
        preset_name: str,
        log=            False,  # log / lin scale
        num_samples=    300,
        devices=        [0,1]*5):

    seed = [1/num_samples * x for x in range(num_samples)]
    if log:
        rng = [math.log10(rng[0]), math.log10(rng[1])]
    params = [rng[0] + (rng[1]-rng[0]) * s for s in seed]
    if log:
        params = [10 ** p for p in params]

    tasks = [{
        'preset_name':  preset_name,
        param:          p} for p in params]

    ompr = OMPRunnerGPU(
        devices=    devices,
        rw_class=   EvalTrainer,
        verb=       1)

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

    eval_base = {
        'param':    'iLR',
        'rng':      [1e-6, 1e-0],
        'log':      True}

    for preset in [
        'use_base_U1',
        'use_one_hidden',
        'use_hidden_stack']:

        ed = {}
        ed.update(eval_base)
        ed['preset_name'] = preset

        evaluate_param(**ed)