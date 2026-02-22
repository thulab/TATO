#!/usr/bin/env python3
"""
Example script demonstrating the new configuration-driven ModelFactory.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model_factory import ModelFactory
import argparse

def main():
    """Demonstrate the new ModelFactory features."""
    
    print("=" * 60)
    print("ModelFactory Configuration-Driven Example")
    print("=" * 60)
    
    # Example 1: List all available models
    print("\n1. Listing all available models:")
    available_models = ModelFactory.get_available_models()
    for i, model_name in enumerate(available_models, 1):
        print(f"   {i}. {model_name}")
    
    # Example 2: Get detailed information about a model
    print("\n2. Getting detailed model information:")
    model_name = "Timer-LOTSA"
    try:
        model_info = ModelFactory.get_model_info(model_name)
        print(f"   Model: {model_name}")
        print(f"   - Class: {model_info['class']}")
        print(f"   - Checkpoint: {model_info['ckpt_path']}")
        print(f"   - Description: {model_info.get('description', 'N/A')}")
        print(f"   - Requires args: {model_info.get('requires_args', False)}")
    except ValueError as e:
        print(f"   Error: {e}")
    
    # Example 3: Load a model (simulated - won't actually load without checkpoints)
    print("\n3. Loading a model (simulated):")
    try:
        # Create dummy arguments for models that require them
        args = argparse.Namespace(
            seed=42,
            batch_size=32,
            learning_rate=1e-4,
            task_name='large_finetune'
        )
        
        # Try to load the model
        print(f"   Attempting to load '{model_name}'...")
        # Note: This will fail if checkpoints don't exist, but shows the API
        # model = ModelFactory.load_model(model_name, "cpu", args)
        # print(f"   Successfully loaded: {model.model_name}")
        print(f"   (Checkpoint loading would happen here if files exist)")
        
    except Exception as e:
        print(f"   Error loading model: {e}")
        print(f"   (This is expected if checkpoints don't exist)")
    
    # Example 4: Validate configuration
    print("\n4. Validating configuration:")
    try:
        ModelFactory.validate_config()
        print("   ✅ Configuration is valid")
    except ValueError as e:
        print(f"   ❌ Configuration validation failed: {e}")
    
    # Example 5: Adding a new model via configuration
    print("\n5. Adding a new model (configuration example):")
    print("""
   To add a new model, simply update configs/model_config.yaml:
   
   NewModel-Example:
     class: Timer  # or any other model class
     ckpt_path: CKPT/new_model_checkpoint
     description: "New example model"
     requires_args: true
   
   Then update model_classes if needed:
   
   model_classes:
     Timer: "model.model_factory.Timer"
     # ... existing classes
     NewModelClass: "path.to.new_model_class"
   """)
    
    # Example 6: Programmatic model discovery
    print("\n6. Programmatic model discovery:")
    print("   You can programmatically discover and work with models:")
    print("""
   # Get all Timer models
   timer_models = [name for name in available_models if 'Timer' in name]
   print(f"Timer models: {timer_models}")
   
   # Get models that require args
   models_needing_args = []
   for model_name in available_models:
       info = ModelFactory.get_model_info(model_name)
       if info.get('requires_args', False):
           models_needing_args.append(model_name)
   print(f"Models requiring args: {models_needing_args}")
   """)
    
    print("\n" + "=" * 60)
    print("Configuration-Driven ModelFactory Benefits:")
    print("=" * 60)
    print("""
1. **Centralized Configuration**: All model definitions in one YAML file
2. **Easy Maintenance**: Add/remove models without code changes
3. **Runtime Discovery**: Programmatically query available models
4. **Validation**: Automatic configuration validation
5. **Flexibility**: Support for different model classes and requirements
6. **Backward Compatible**: Existing code continues to work
    """)
    
    print("\nUsage in existing code remains the same:")
    print("""
from model.model_factory import ModelFactory
import argparse

args = argparse.Namespace()
model = ModelFactory.load_model('Timer-LOTSA', 'cuda:0', args)
# ... use model as before
    """)

if __name__ == "__main__":
    main()
