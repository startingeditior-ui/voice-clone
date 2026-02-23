import os
import sys

# Ensure the RVC directory is in path
RVC_ROOT = os.path.join(os.getcwd(), "Retrieval-based-Voice-Conversion-WebUI")
sys.path.append(RVC_ROOT)

# Save current root and change to RVC root for path sensitive relative imports
PROJECT_ROOT = os.getcwd()
os.chdir(RVC_ROOT)

try:
    from rvc_wrapper import RVCWrapper
    print(f"Successfully imported RVCWrapper. Current CWD: {os.getcwd()}")
    
    # Initialize
    rvc = RVCWrapper()
    print("Successfully initialized RVCWrapper")
    
    # Check for models
    weights_dir = os.path.join(RVC_ROOT, "assets", "weights")
    models = [f for f in os.listdir(weights_dir) if f.endswith(".pth")]
    
    if not models:
        print("No models found in RVC weights directory.")
    else:
        model_name = models[0]
        print(f"Loading model: {model_name}")
        rvc.load_model(model_name)
        print("Model loaded successfully")
        
        # Test conversion if recorded.wav exists
        source = os.path.join(PROJECT_ROOT, "input", "recorded.wav")
        output = os.path.join(PROJECT_ROOT, "output", "test_mimic_rvc_env.wav")
        
        if os.path.exists(source):
            print(f"Converting {source}...")
            # Use 'rmvpe' as it's common
            rvc.convert(source, output, f0_method="rmvpe")
            print(f"Conversion successful! Output at {output}")
        else:
            print("input/recorded.wav not found, skipping conversion test.")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error: {e}")
finally:
    # Always change back to project root
    os.chdir(PROJECT_ROOT)
