import os


class TransformationFactory(object):
    __transformation_dict = {}
    builtin_loaded = False
    root_directory = os.path.dirname(os.path.dirname(__file__))
    library_directory = os.path.join(root_directory, 'transformation', 'library')
    
    @staticmethod
    def get_transformation_dict():
        if not TransformationFactory.builtin_loaded:
            TransformationFactory.load_builtin_transformations()
        return TransformationFactory.__transformation_dict
    
    @staticmethod
    def load_builtin_transformations():
        directory_path = TransformationFactory.library_directory
        all_transformation_classes = TransformationFactory.__define_classes_in_dir(directory_path)
        TransformationFactory.__transformation_dict.update(all_transformation_classes)
        TransformationFactory.builtin_loaded = True
    
    def __define_classes_in_dir(directory_path):
        all_classes = {}
        for dirpath, dirnames, filenames in os.walk(directory_path):
            # filter out non-python files and __init__.py
            filenames = [f for f in filenames if f.endswith('.py') and f != '__init__.py']
            # filter out __pycache__ directories
            if '__pycache__' in dirpath:
                continue
            # Dynamically load all model classes from the relative model directory
            relative_path = os.path.relpath(dirpath, TransformationFactory.root_directory)
            module_path_prefix = relative_path.replace('/', '.').replace('\\', '.').lstrip('.')
            try:
                classes = TransformationFactory.__import_classes_from_module(module_path_prefix, filenames)
                print(f"classes: {classes}")
            except ModuleNotFoundError as e:
                # print log
                print(f"module_path_prefix: {module_path_prefix}, {','.join(filenames)}")
            all_classes.update(classes)
        return all_classes
        
    def __import_classes_from_module(module_path_prefix, filenames):
        import importlib
        classes = {}
        for filename in filenames:
            module_name = filename[:-3]  # remove .py extension
            # import pdb; pdb.set_trace()
            full_module_path = f"{module_path_prefix}.{module_name}"
            module = importlib.import_module(full_module_path)
            model_found = False
            for attr_name in dir(module):
                if attr_name == 'Transformation':
                    model_found = True
                    break
            if model_found:
                classes[module_name] = module
        return classes
    
# if __name__ == '__main__':
#     transformation_dict = TransformationFactory.get_transformation_dict()
#     print(transformation_dict)