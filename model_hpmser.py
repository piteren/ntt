from ptools.pms.hpmser.search_function import hpmser_GX

from model_presets import get_preset
from model_train import train_model


if __name__ == '__main__':

    presets_dm = [
        #('use_base_U0',         10),
        #('use_base_U1',         10),
        #('use_base_U2',         10),
        #('use_one_hidden',      10),
        #('use_hidden_stack',    10),
        #('use_hidden_stack_bd', 10),
        #('use_drt',             10),
        #('use_more',            6),
        #('seq_reduced',         10),
        #('seq_cnn',             2),
        #('seq_cnn_tf',          2),
        #('seq_cnn_ind',         2),
        #('seq_cnn_tf_layDRTEX', 2),
        #('seq_tns',             4),
        ('seq_tns_tf',          3),
        #('seq_tns_ind',         3),
        #('seq_tat',             8),
        #('seq_tat_tf',          5),
        #('seq_tat_ind',         8),
    ]

    for preset_name, dm in presets_dm:
        hpmser_GX(
            func=           train_model,
            psdd=           get_preset(preset_name).pop('psdd'),
            func_defaults=  {'preset_name':preset_name},
            name=           f'hpmser_for_{preset_name}',
            #devices=        [0,1]*dm,
            devices=        [0]*dm*2,
            #config_upd=     100,#None,
            #n_loops=        2000,
            verb=           1)