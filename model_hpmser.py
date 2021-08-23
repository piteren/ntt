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

    preset_name = 'use_base'
    #preset_name = 'use_hidden'
    #preset_name = 'use_drt'

    preset = get_preset(preset_name)

    func_defaults = {
        'fwd_func':     preset.pop('fwd_func'),
        'model_type':   preset.pop('model_type')}
    psdd = preset.pop('psdd')
    func_defaults.update(preset)

    hpmser_GX(
        func=           train_for_score,
        func_defaults=  func_defaults,
        psdd=           psdd,
        devices=        [0,1]*4,
        verb=           1)
