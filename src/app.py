from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import sys

# Ensure the project root is on the path so src.* imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Monkeypatch torch.load for compatibility with PyTorch 2.4+ (WeightsUnpickler error)
import torch
orig_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return orig_load(*args, **kwargs)
torch.load = patched_load

import shutil
import uuid

# Workaround for lazy-loading ModuleNotFoundError: Could not import module 'GPT2PreTrainedModel'
try:
    from transformers import GenerationMixin, GPT2PreTrainedModel
    print("Pre-loaded transformers successfully.")
except ImportError:
    print("Warning: Failed to pre-load transformers. This might be expected if handled by TTS.")

from src.transcribe import transcribe_audio
from src.synthesize import synthesize, mimic_voice

app = FastAPI()

# Initialize RVC Wrapper (Lazy Loading)
rvc_instance = None
def get_rvc():
    global rvc_instance
    if rvc_instance is None:
        print("Initializing RVC Wrapper...")
        from src.rvc_wrapper import RVCWrapper
        rvc_instance = RVCWrapper()
    return rvc_instance

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(PROJECT_ROOT, "input")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Serve static files for audio access
app.mount("/static/output", StaticFiles(directory=OUTPUT_DIR), name="output")

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    template_path = os.path.join(PROJECT_ROOT, "frontend", "index.html")
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Welcome to Voice Clone Project</h1><p>UI template not found. Backend is running.</p>"

@app.post("/process-voice")
async def process_voice(
    file: UploadFile = File(...),
    accent: str = Form("neutral"),
    tone: str = Form("calm"),
    custom_text: str = Form(None)
):
    try:
        # Save uploaded file with a unique name
        file_id = str(uuid.uuid4())
        input_filename = f"{file_id}_{file.filename}"
        input_path = os.path.join(UPLOAD_DIR, input_filename)

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 1. Get Text (Custom or Transcribed)
        if custom_text and custom_text.strip():
            print(f"Using custom text: {custom_text}")
            text = custom_text
        else:
            print(f"Transcribing {input_path}...")
            text = transcribe_audio(input_path)
            if not text.strip():
                text = "Hello, this is a test voice."

        # 2. Synthesize
        output_filename = f"{file_id}_output.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        print(f"Synthesizing to {output_path} with accent={accent}, tone={tone}...")
        synthesize(text, input_path, accent=accent, tone=tone, output_path=output_path)

        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Synthesis failed to produce output file")

        return {
            "text": text,
            "output_url": f"/audio/output/{output_filename}"
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in process_voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/mimic-voice")
async def mimic_voice_endpoint(
    target_voice_file: UploadFile = File(...),
    performance_file: UploadFile = File(None),
    custom_text: str = Form(None)
):
    """
    Smart Mimic: accepts EITHER custom_text OR performance_file (or both).
    - custom_text provided → synthesize it directly with XTTS using target voice (fast)
    - performance_file provided → transcribe it first, then synthesize with target voice
    - Both provided → custom_text takes priority (skip transcription)
    """
    try:
        file_id = str(uuid.uuid4())

        # Save target voice audio (the person we want to sound like)
        target_filename = f"{file_id}_target_{target_voice_file.filename}"
        target_path = os.path.join(UPLOAD_DIR, target_filename)
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(target_voice_file.file, buffer)

        output_filename = f"{file_id}_smart_mimic.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        text_used = custom_text.strip() if custom_text and custom_text.strip() else None

        if text_used:
            # Fast path: use typed text directly — no transcription needed
            print(f"[Smart Mimic] Using custom text: {text_used[:80]}..." if len(text_used) > 80 else f"[Smart Mimic] Using custom text: {text_used}")
            from src.synthesize import tts as _tts
            _tts.tts_to_file(
                text=text_used,
                speaker_wav=target_path,
                language="en",
                file_path=output_path
            )
            transcribed_text = text_used

        elif performance_file and performance_file.filename:
            # Transcription path: transcribe performance, then synthesize
            perf_filename = f"{file_id}_perf_{performance_file.filename}"
            perf_path = os.path.join(UPLOAD_DIR, perf_filename)
            with open(perf_path, "wb") as buffer:
                shutil.copyfileobj(performance_file.file, buffer)

            print(f"[Smart Mimic] Transcribing performance: {perf_path}")
            _, transcribed_text = mimic_voice(
                performance_wav=perf_path,
                target_voice_wav=target_path,
                output_path=output_path
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either custom_text or a performance_file to process."
            )

        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Smart Mimic failed to produce output file")

        return {
            "output_url": f"/audio/output/{output_filename}",
            "transcribed_text": transcribed_text
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audio/output/{filename}")
async def get_audio(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/wav")
    return JSONResponse(status_code=404, content={"message": "File not found"})

@app.post("/convert-voice")
async def convert_voice(
    performance_file: UploadFile = File(...),
    target_voice_file: UploadFile = File(...),
    pitch_shift: int = Form(0),
    index_rate: float = Form(0.75),
    filter_radius: int = Form(3),
    rms_mix_rate: float = Form(0.25),
    protect: float = Form(0.33)
):
    try:
        file_id = str(uuid.uuid4())

        # 1. Save Performance File (the one with the accent/tone)
        perf_filename = f"{file_id}_perf_{performance_file.filename}"
        perf_path = os.path.join(UPLOAD_DIR, perf_filename)
        with open(perf_path, "wb") as buffer:
            shutil.copyfileobj(performance_file.file, buffer)

        # 2. Save Target Voice File (the identity we want to clone)
        target_filename = f"{file_id}_target_{target_voice_file.filename}"
        target_path = os.path.join(UPLOAD_DIR, target_filename)
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(target_voice_file.file, buffer)

        # Check if any .pth models exist in weights
        weights_dir = os.path.join(PROJECT_ROOT, "Retrieval-based-Voice-Conversion-WebUI", "assets", "weights")
        models = [f for f in os.listdir(weights_dir) if f.endswith(".pth")]

        if not models:
            raise HTTPException(
                status_code=400,
                detail="Mimicry mode requires an RVC model (.pth) in 'Retrieval-based-Voice-Conversion-WebUI/assets/weights'. Use Text-to-Speech mode for standard voice cloning."
            )

        # Use the first available model
        model_name = models[0]
        rvc = get_rvc()
        rvc.load_model(model_name)

        output_filename = f"{file_id}_mimic_output.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        rvc.convert(
            source_path=perf_path,
            output_path=output_path,
            f0_up_key=pitch_shift,
            index_rate=index_rate,
            filter_radius=filter_radius,
            rms_mix_rate=rms_mix_rate,
            protect=protect
        )

        return {
            "output_url": f"/audio/output/{output_filename}",
            "model_used": model_name
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting server...")
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=False)
