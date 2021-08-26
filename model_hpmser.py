from ptools.pms.hpmser.search_function import hpmser_GX

from model_presets import get_preset
from model_train import train_model


if __name__ == '__main__':

    presets = [
        #'use_base_U0',
        #'use_base_U1',
        #'use_one_hidden',
        'use_hidden_stack',
        #'use_drt',
    ]

    for preset_name in presets:
        hpmser_GX(
            func=           train_model,
            psdd=           get_preset(preset_name).pop('psdd'),
            func_defaults=  {'preset_name':preset_name},
            name=           f'hpmser_for_{preset_name}',
            devices=        [0,1]*10,
            adv_config_upd= 1000,
            n_loops=        2000,
            verb=           1)