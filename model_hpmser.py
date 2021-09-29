from ptools.pms.hpmser.search_function import hpmser_GX

from model_presets import get_preset
from model_train import train_model


if __name__ == '__main__':

    presets_dm = [
        #('use_base_U0',         10),
        #('use_base_U1',         10),
        #('use_one_hidden',      10),
        #('use_hidden_stack',    10),
        #('use_drt',             10),
        #('use_more',            10),
        #('seq_reduced',         10),
        ('seq_cnn',             2),
        #('seq_tns',             5),
        #('seq_tat',             10),
    ]

    for preset_name, dm in presets_dm:
        hpmser_GX(
            func=           train_model,
            psdd=           get_preset(preset_name).pop('psdd'),
            func_defaults=  {'preset_name':preset_name},
            name=           f'hpmser_for_{preset_name}',
            devices=        [0]*dm*2,
            #devices=        [0,1]*dm,
            #config_upd=     100,#None,
            #n_loops=        2000,
            verb=           1)