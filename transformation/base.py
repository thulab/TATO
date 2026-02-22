from typing import Any, Dict
import numpy as np


class BaseTransformation:
    """Abstract base class for all data transformations in TATO.
    
    This class defines the interface that all transformation classes must implement.
    Transformations are used to preprocess time series data before feeding it to
    foundation models, and to post-process model outputs back to the original scale.
    
    Attributes:
        search_space: Dictionary defining the hyperparameter search space for the
                     transformation. Used by the tuner to optimize transformation
                     parameters.
    """
    
    search_space: Dict[str, Any] = {}
    
    def __init__(self, **kwargs):
        """Initialize the transformation with given parameters.
        
        Args:
            **kwargs: Transformation-specific parameters. Subclasses should
                     validate and store these parameters.
        """
        pass

    def pre_process(self, data: np.ndarray) -> np.ndarray:
        """Apply transformation to input data.
        
        This method transforms the input data before feeding it to the model.
        It should be implemented by all subclasses.
        
        Args:
            data: Input time series data. Shape depends on the transformation
                  (typically (batch_size, seq_len, num_features) or variations).
            
        Returns:
            Transformed data ready for model input.
            
        Raises:
            NotImplementedError: If subclass doesn't implement this method.
        """
        raise NotImplementedError

    def post_process(self, data: np.ndarray) -> np.ndarray:
        """Reverse transformation on model output.
        
        This method applies the inverse transformation to model predictions,
        bringing them back to the original data scale.
        
        Args:
            data: Model output/predictions that need to be transformed back
                  to original scale.
            
        Returns:
            Data transformed back to original scale.
            
        Raises:
            NotImplementedError: If subclass doesn't implement this method.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'BaseTransformation':
        """Create transformation instance from configuration dictionary.
        
        This factory method allows creating transformation instances from
        configuration dictionaries, which is useful for loading transformations
        from saved configurations.
        
        Args:
            config: Dictionary containing transformation parameters.
            
        Returns:
            Instance of the transformation class initialized with the given
            configuration.
        """
        return cls(**config)
