from transcribe import transcribe_audio
import os

# Use an existing sample file
test_input = "input/recorded.wav"
if not os.path.exists(test_input):
    # Try another one if recorded.wav is missing
    test_input = "input/vishal.wav"

if os.path.exists(test_input):
    print(f"Testing transcription for: {test_input}")
    try:
        text = transcribe_audio(test_input)
        print(f"Transcription successful!")
        print(f"Result: {text}")
    except Exception as e:
        print(f"Transcription failed: {e}")
else:
    print(f"No test file found at {test_input}")
