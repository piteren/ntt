from typing import List, Any

from ptools.mpython.omp import OMPRunnerGPU, RunningWorkerGPU

from model_train import train_model


class NNTTrainer(RunningWorkerGPU):

    def run(self, preset_name) -> Any:
        return train_model(
            preset_name=    preset_name,
            devices=        self.devices,
            verb=           0)


def accumulated_TRTS(
        presets: List[str],
        devices=        [0,1]*2,
        num_acc_runs=   10) -> dict:

    ompr = OMPRunnerGPU(
        devices=    devices,
        rw_class=   NNTTrainer,
        verb=       1)

    acc_test_results = {}

    for preset_name in presets:
        tasks = [{'preset_name':preset_name}] * num_acc_runs
        results = ompr.process(tasks, exit_after=False)
        acc_test_results[preset_name] = results

    ompr.exit()

    print('Accumulated TRTS results:')
    for k in acc_test_results:
        print(f'model {k} - {sum(acc_test_results[k])/num_acc_runs:.4f}')
    return acc_test_results


if __name__ == '__main__':

    presets = [
        'use_base_U0',
        'use_base_U1',
        #'use_one_hidden'
        #'use_hidden_stack',
        #'use_drt',
        #'seq'
    ]

    accumulated_TRTS(
        presets=    presets,
        #devices=    None
    )
