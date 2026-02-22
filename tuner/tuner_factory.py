from transformation.transformation_factory import TransformationFactory
from optuna.distributions import CategoricalDistribution, IntDistribution
from tuner.base import Tuner
from math import ceil

multi_search = {
    'trimmer':False, 'sampler':False, 'inputer':False, 'denoiser':True, 
    'warper':True, 'differentiator':False, 'normalizer':True, 'aligner':False
}

class TunerFactory(object):
    
    @staticmethod
    def build_search_space(transformation_names, patch_len = 96):
        distribution = {}
        distribution['inference_mode'] = CategoricalDistribution(['infer0', 'infer1'])
        distribution['clip_factor'] = CategoricalDistribution(['none', '0', '0.25'])
        transformation_dict = TransformationFactory.get_transformation_dict()
        for key in transformation_names:
            value = transformation_dict[key]
            for param_key, param_value in value.Transformation.search_space.items():
                search_key = TunerFactory.get_search_key(key, param_key)
                # TODO Check rightness
                if all(isinstance(x, str) for x in param_value) or len(param_value) == 1:
                    distribution[search_key] = CategoricalDistribution(param_value)
                else:
                    step = ceil(param_value[1] - param_value[0])
                    distribution[search_key] = IntDistribution(min(param_value), max(param_value), step=step)
        return distribution
    
    @staticmethod
    def build_mul_search_space(transformation_names, patch_len = 96, f_dim = 1):
        distribution = {}
        distribution['inference_mode'] = CategoricalDistribution(['infer0', 'infer1'])
        distribution['clip_factor'] = CategoricalDistribution(['none', '0', '0.25'])
        transformation_dict = TransformationFactory.get_transformation_dict()
        for key in transformation_names:
            value = transformation_dict[key]
            for param_key, param_value in value.Multi_Transformation.search_space.items():
                if param_key != "method" or not multi_search[key]:
                    search_key = TunerFactory.get_search_key(key, param_key)
                    # TODO Check rightness
                    if all(isinstance(x, str) for x in param_value) or len(param_value) == 1:
                        distribution[search_key] = CategoricalDistribution(param_value)
                    else:
                        step = ceil(param_value[1] - param_value[0])
                        distribution[search_key] = IntDistribution(min(param_value), max(param_value), step=step)
                else:
                    for i in range(f_dim):
                        search_key = f'f{i}_{TunerFactory.get_search_key(key, param_key)}'
                        if all(isinstance(x, str) for x in param_value) or len(param_value) == 1:
                            distribution[search_key] = CategoricalDistribution(param_value)
                        else:
                            step = ceil(param_value[1] - param_value[0])
                            distribution[search_key] = IntDistribution(min(param_value), max(param_value), step=step)
        return distribution
    
    @staticmethod
    def get_vanilla_params(transformation_names, f_dim=1):
        vanilla_params = {
            'inference_mode': 'infer1',
            'clip_factor': 'none',
            'trimmer_seq_l': 7,
            'inputer_detect_method': 'none',
            'inputer_fill_method': 'linear_interpolate',
            # 'denoiser_method': 'none',
            # 'warper_method': 'none',
            'differentiator_n': 0,
            # 'normalizer_method': 'none',
            'normalizer_mode': 'input',
            'sampler_factor': 1,
            'aligner_mode': 'none',
            'aligner_method': 'edge_pad'
        }
        transformation_dict = TransformationFactory.get_transformation_dict()
        for key in transformation_names:
            if multi_search[key]:
                value = transformation_dict[key]
                for i in range(f_dim):
                    search_key = f'f{i}_{TunerFactory.get_search_key(key, "method")}'
                    vanilla_params[search_key] = value.Multi_Transformation.search_space['method'][0]
        return vanilla_params

    @staticmethod
    def get_search_key(transformation_name, param_name):
        return transformation_name + '_' + param_name

    @staticmethod
    def build_optuna_tuner(direction='minimize', enqueue_param_dicts=None, mode=None, seed=0):
        tuner = Tuner(direction, enqueue_param_dicts=enqueue_param_dicts, mode=mode, seed=seed)
        return tuner
