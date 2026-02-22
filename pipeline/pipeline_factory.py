from httpx import patch
from transformation.transformation_factory import TransformationFactory
from pipeline.base import BasePipeline
from tuner.tuner_factory import TunerFactory, multi_search
from math import ceil


class PipelineFactory(object):
    
    infer_mode_0 = ['trimmer', 'inputer', 'denoiser', 'warper', 'differentiator', 'normalizer', 'sampler', 'aligner']
    infer_mode_1 = ['trimmer', 'sampler', 'inputer', 'denoiser', 'warper', 'differentiator', 'normalizer', 'aligner']

    @staticmethod
    def build_trial_pipeline_by_transformation_names(trial, model, transformation_names, configs, aug=None, mode='train', id=None, save_dir=None, plt=False) -> BasePipeline: 
        transformation_dict = TransformationFactory.get_transformation_dict()
        transformation_list = {}
        pred_true = configs.get('pred_len', 96)
        patch_len = configs.get('patch_len', 96)
        for key in transformation_names:
            value = transformation_dict[key]
            init_param_dict = {}
            init_param_dict.update(configs)
            init_param_dict["clip_factor"] = trial.params[TunerFactory.get_search_key('clip', 'factor')]
            for param_key in value.Transformation.search_space.keys():
                init_param_dict[param_key] = trial.params[TunerFactory.get_search_key(key, param_key)]
                # print(f"Transformation {key}: setting param {param_key} = {init_param_dict[param_key]}")
                if TunerFactory.get_search_key(key, param_key) == "sampler_factor":
                    pred_true = ceil(pred_true / init_param_dict["factor"] / patch_len) * patch_len
            try:
                transformation = [key, value.Transformation(**init_param_dict)]
            except Exception as e:
                print(f"Error when initializing transformation {key} with params {init_param_dict}: {e}")
                raise e
            transformation_list.update({transformation[0]: transformation[1]})
        
        true_transformation_list = []

        infer_mode = trial.params['inference_mode']
        if infer_mode == 'infer0':
            for key in PipelineFactory.infer_mode_0:
                if key in transformation_names:
                    true_transformation_list.append(transformation_list[key])
        elif infer_mode == 'infer1':
            for key in PipelineFactory.infer_mode_1:
                if key in transformation_names:
                    true_transformation_list.append(transformation_list[key])
        else:
            raise ValueError(f"Invalid inference mode: {infer_mode}")
        
        return BasePipeline(transformations=true_transformation_list, config=None, model=model, pred_len=pred_true, augmentor=aug, mode=mode, id=id, save_dir=save_dir, plt=plt)
    

    @staticmethod
    def build_mul_pipeline_by_transformation_names(trial, model, transformation_names, configs, aug=None, mode='train', id=None, save_dir=None, plt=False) -> BasePipeline: 
        transformation_dict = TransformationFactory.get_transformation_dict()
        transformation_list = {}
        pred_true = configs.get('pred_len', 96)
        patch_len = configs.get('patch_len', 96)
        f_dim = configs.get('feature_dim', 1)
        for key in transformation_names:
            value = transformation_dict[key]
            init_param_dict = {}
            init_param_dict.update(configs)
            init_param_dict["clip_factor"] = trial.params[TunerFactory.get_search_key('clip', 'factor')]
            for param_key in value.Multi_Transformation.search_space.keys():
                if multi_search[key] and param_key == "method":
                    param_list = []
                    for i in range(f_dim):
                        param_list.append(trial.params[f'f{i}_{key}_{param_key}'])
                    init_param_dict['method_list'] = param_list
                else:
                    init_param_dict[param_key] = trial.params[TunerFactory.get_search_key(key, param_key)]
                if TunerFactory.get_search_key(key, param_key) == "sampler_factor":
                    pred_true = ceil(pred_true / init_param_dict["factor"] / patch_len) * patch_len
            try:
                transformation = [key, value.Multi_Transformation(**init_param_dict)]
            except Exception as e:
                print(f"Error when initializing transformation {key} with params {init_param_dict}: {e}")
                raise e
            transformation_list.update({transformation[0]: transformation[1]})
        
        true_transformation_list = []

        infer_mode = trial.params['inference_mode']
        if infer_mode == 'infer0':
            for key in PipelineFactory.infer_mode_0:
                if key in transformation_names:
                    true_transformation_list.append(transformation_list[key])
        elif infer_mode == 'infer1':
            for key in PipelineFactory.infer_mode_1:
                if key in transformation_names:
                    true_transformation_list.append(transformation_list[key])
        else:
            raise ValueError(f"Invalid inference mode: {infer_mode}")
        
        return BasePipeline(transformations=true_transformation_list, config=None, model=model, pred_len=pred_true, augmentor=aug, mode=mode, id=id, save_dir=save_dir, plt=plt)
