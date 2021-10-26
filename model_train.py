from typing import List

from ptools.mpython.mpdecor import proc

from model_presets import get_preset


# single train and test of model
def train_model(
        preset_name: str,
        hpmser_mode=        False,
        save_wt=            True,
        devices=            -1,
        verb=               1,
        **kwargs):
    preset = get_preset(preset_name)
    model_type = preset.pop('model_type')
    preset.update(kwargs)
    preset['verb'] = verb-1
    model = model_type(
        devices=            devices,
        hpmser_mode=        hpmser_mode,
        name_timestamp=     True,
        silent_TF_warnings= True,
        **preset)
    result = model.train(save=save_wt)
    if verb>0: print(f'model {preset_name} resulted: {result:.4f}')
    return result

@proc # single non-blocking train_model, does not return anything
def train_NB(**kwargs):
    train_model(**kwargs)

def train_many(presets: List[str]):
    for dx, preset_name in enumerate(presets):
        print(f'training of {preset_name} started..')
        train_NB(
            preset_name=    preset_name,
            save_wt=        False,
            devices=        dx%2)


if __name__ == '__main__':

    train_model(preset_name='seq_cnn_tf', verb=2)

    presets = [
        'use_base_U0',
        'use_base_U1',
        'use_base_U2',
        'use_one_hidden',
        'use_hidden_stack',
        'seq_cnn_tf',
        'seq_cnn_ind',
        'seq_tns_tf',
        'seq_tns_ind',
        'seq_tat',
        'seq_tat_tf',
    ]
    #train_many(presets)