from ptools.pms.hpmser.search_function import hpmser_GX

from model_presets import get_preset


def train_for_score(
        model_type,
        fwd_func,
        devices,
        **kwargs):
    model = model_type(
        fwd_func=       fwd_func,
        mdict=          kwargs,
        devices=        devices)
    return model.train()



if __name__ == '__main__':

    presets = [
        'use_base_U1',
        #'use_one_hidden'
        #'use_hidden_stack',
        #'use_drt',
        #'seq'
    ]

    for preset_name in presets:

        preset = get_preset(preset_name)

        func_defaults = {
            'fwd_func':     preset.pop('fwd_func'),
            'model_type':   preset.pop('model_type')}
        psdd = preset.pop('psdd')
        func_defaults.update(preset)

        hpmser_GX(
            func=           train_for_score,
            psdd=           psdd,
            func_defaults=  func_defaults,
            name=           f'hpmser_for_{preset_name}',
            devices=        [0,1]*8,
            n_loops=        1000,
            verb=           1)
