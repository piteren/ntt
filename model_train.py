from model_presets import get_preset


if __name__ == '__main__':

    #preset_name = 'use_hidden'
    preset_name = 'seq'

    preset = get_preset(preset_name)
    fwd_func = preset.pop('fwd_func')
    model_type = preset.pop('model_type')

    model = model_type(
        fwd_func=       fwd_func,
        mdict=          preset,
        name_timestamp= True,
        verb=           1)
    print(model.train())