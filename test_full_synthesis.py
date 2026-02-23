from synthesize import synthesize
import os

# Ensure input directory exists and has a sample file or use a dummy
dummy_input = "input/recorded.wav"
if not os.path.exists(dummy_input):
    print(f"Warning: {dummy_input} not found. Synthesis might fail.")

try:
    print("Starting test synthesis...")
    # Updated synthesize call with output_path
    synthesize("This is a test transcription.", dummy_input, output_path="output/test_verification.wav")
    print("Synthesis completed successfully! Output: output/test_verification.wav")
except Exception as e:
    print(f"Synthesis failed with error: {e}")
    import traceback
    traceback.print_exc()
