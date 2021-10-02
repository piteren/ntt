from model_presets import get_preset

# single train and test of model
def train_model(
        preset_name: str,
        hpmser_mode=        False,
        devices=            -1,
        return_ts_acc_max=  False,
        verb=               1,
        **kwargs):
    preset = get_preset(preset_name)
    model_type = preset.pop('model_type')
    preset.update(kwargs)
    preset['verb'] = verb
    model = model_type(
        devices=            devices,
        hpmser_mode=        hpmser_mode,
        name_timestamp=     True,
        silent_TF_warnings= True,
        **preset)
    return model.train(return_ts_acc_max=return_ts_acc_max)


if __name__ == '__main__':

    #preset_name = 'use_base_U0'
    #preset_name = 'use_base_U1'
    preset_name = 'use_base_U2'
    #preset_name = 'use_one_hidden'
    #preset_name = 'use_hidden_stack'
    #preset_name = 'use_drt'
    #preset_name = 'seq_tns'

    result = train_model(
        preset_name=    preset_name,
        #devices=        None,
        #hpmser_mode=    True,
    )
    print(result)