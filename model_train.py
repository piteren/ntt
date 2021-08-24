from model_presets import get_preset


def train_model(
        preset_name:str,
        devices=    -1,
        verb=       1):
    preset = get_preset(preset_name)
    fwd_func = preset.pop('fwd_func')
    model_type = preset.pop('model_type')
    if verb==0: preset['hpmser_mode'] = True
    model = model_type(
        fwd_func=       fwd_func,
        mdict=          preset,
        name_timestamp= True,
        devices=        devices,
        verb=           verb)
    return model.train()


if __name__ == '__main__':

    #preset_name = 'use_base'
    preset_name = 'use_hidden'
    #preset_name = 'use_drt'
    #preset_name = 'seq'

    train_model(
        preset_name,
        devices=    None)