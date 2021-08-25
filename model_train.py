from model_presets import get_preset

# single train and test of model
def train_model(
        preset_name:str,
        hpmser_mode=        False,
        devices=            -1,
        return_ts_acc_max=  False,
        save=               True,
        verb=               1,
        **kwargs):
    preset = get_preset(preset_name)
    fwd_func = preset.pop('fwd_func')
    model_type = preset.pop('model_type')
    preset.update(kwargs)
    model = model_type(
        fwd_func=       fwd_func,
        mdict=          preset,
        hpmser_mode=    hpmser_mode,
        name_timestamp= True,
        devices=        devices,
        read_only=      not save,
        do_logfile=     not save,
        verb=           verb)
    return model.train(return_ts_acc_max=return_ts_acc_max)


if __name__ == '__main__':

    preset_name = 'use_base_U0'
    #preset_name = 'use_base_U1'
    #preset_name = 'use_one_hidden'
    #preset_name = 'use_drt'
    #preset_name = 'seq'

    result = train_model(
        preset_name=    preset_name,
        devices=        None,
        #hpmser_mode=    True,
    )
    print(result)