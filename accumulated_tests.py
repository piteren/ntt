import random
from typing import List, Any

from ptools.mpython.omp import OMPRunner, RunningWorker
from ptools.lipytools.stats import msmx

from model_train import train_model


class NNTTrainer(RunningWorker):

    def __init__(self, devices):
        self.devices = devices

    def process(self, preset_name) -> Any:
        return train_model(
            preset_name=        preset_name,
            devices=            self.devices,
            hpmser_mode=        True,
            verb=               0)


# accumulated train and test for given preset
def accumulated_TRTS(
        presets: List[str],
        devices=            [0,1]*5,
        num_acc_runs=       10) -> dict:

    ompr = OMPRunner(
        rw_class=       NNTTrainer,
        rw_lifetime=    1,
        devices=        devices,
        verb=           1)

    acc_test_results = {}
    tasks = []
    for preset_name in presets:
        acc_test_results[preset_name] = []
        tasks += [{'preset_name':preset_name}] * num_acc_runs
    random.shuffle(tasks) # shuffle tasks for balanced load

    all_results = ompr.process(tasks)

    for td, res in zip(tasks,all_results):
        acc_test_results[td['preset_name']].append(res)

    print(f'Accumulated ({num_acc_runs}) TRTS results:')
    for k in acc_test_results:
        stats = msmx(acc_test_results[k])
        print(f'model: {k:25s} accuracy: mean:{stats["mean"]:.4f}, std:{stats["std"]:.4f}')
    return acc_test_results


if __name__ == '__main__':

    accumulated_TRTS(
        devices=[0]*5,
        presets= [
            #'use_base_U0',
            #'use_base_U1',
            #'use_base_U2',
            #'use_one_hidden',
            #'use_hidden_stack',
            #'seq_reduced',
            'seq_cnn',
            #'seq_cnn_tf',
            #'seq_cnn_ind',
            #'seq_cnn_tf_lay_DRT',
            #'seq_tns',
            #'seq_tns_tf',
            #'seq_tns_ind',
            #'seq_tat',
            #'seq_tat_tf',
            #'seq_tat_ind',
            #'seq_tat_tf_NEW', # 0.7998
            #'seq_tat_ind_NEW',# 0.8004
    ])
