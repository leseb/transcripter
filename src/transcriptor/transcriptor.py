import os
import math
import tempfile
import logging
import json
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from pydub import AudioSegment
from openai import OpenAI
import aiofiles

# Configure logging
log_format = (
    "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[logging.FileHandler("transcriptor.log"), logging.StreamHandler()],
)

# Set specific loggers
logging.getLogger("uvicorn").setLevel(logging.INFO)
logging.getLogger("fastapi").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="src/transcriptor/templates")

# Initialize OpenAI client with logging
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY environment variable is not set!")
    raise ValueError("OPENAI_API_KEY environment variable is required")

client = OpenAI(api_key=api_key)
logger.info("OpenAI client initialized successfully")

MAX_CHUNK_SECONDS = 1200  # Reduced to ensure we stay under 25MB limit
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25MB limit for OpenAI
logger.info(f"Maximum chunk size set to {MAX_CHUNK_SECONDS} seconds")

# Transcript storage configuration
TRANSCRIPTS_DIR = Path("transcripts")
TRANSCRIPTS_DIR.mkdir(exist_ok=True)
METADATA_FILE = TRANSCRIPTS_DIR / "metadata.json"

# Global progress tracking
transcription_progress = {}


def load_transcript_metadata():
    """Load transcript metadata from file"""
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading metadata: {e}")
            return {}
    return {}


def save_transcript_metadata(metadata):
    """Save transcript metadata to file"""
    try:
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info("Transcript metadata saved successfully")
    except IOError as e:
        logger.error(f"Error saving metadata: {e}")


def add_transcript_to_metadata(
    filename, original_filename, duration, processing_time, transcript_length
):
    """Add a new transcript to the metadata"""
    metadata = load_transcript_metadata()
    transcript_id = filename.replace(".txt", "")

    metadata[transcript_id] = {
        "filename": filename,
        "original_filename": original_filename,
        "created_at": datetime.now().isoformat(),
        "duration": duration,
        "processing_time": processing_time,
        "transcript_length": transcript_length,
        "file_path": str(TRANSCRIPTS_DIR / filename),
    }

    save_transcript_metadata(metadata)
    return metadata


def get_available_transcripts():
    """Get list of all available transcripts with metadata"""
    metadata = load_transcript_metadata()
    transcripts = []

    for transcript_id, info in metadata.items():
        # Check if file still exists
        file_path = Path(info["file_path"])
        if file_path.exists():
            transcripts.append(
                {
                    "id": transcript_id,
                    "filename": info["filename"],
                    "original_filename": info["original_filename"],
                    "created_at": info["created_at"],
                    "duration": info["duration"],
                    "processing_time": info["processing_time"],
                    "transcript_length": info["transcript_length"],
                }
            )
        else:
            logger.warning(f"Transcript file not found: {file_path}")

    # Sort by creation date (newest first)
    transcripts.sort(key=lambda x: x["created_at"], reverse=True)
    return transcripts


def get_optimal_chunk_size(audio_duration_ms, target_size_bytes=20 * 1024 * 1024):
    """Calculate optimal chunk size to stay under file size limit"""
    # Estimate bytes per second based on 16kHz mono WAV
    # 16kHz * 2 bytes per sample * 1 channel = 32,000 bytes per second
    bytes_per_second = 32000

    # Calculate max duration for target size
    max_duration_seconds = target_size_bytes / bytes_per_second

    # Convert to milliseconds and apply some safety margin (80% of limit)
    max_duration_ms = int(max_duration_seconds * 1000 * 0.8)

    # Don't exceed our hard limit
    max_duration_ms = min(max_duration_ms, MAX_CHUNK_SECONDS * 1000)

    logger.info(
        f"Calculated optimal chunk size: {max_duration_ms}ms ({max_duration_ms / 1000:.1f}s)"
    )
    return max_duration_ms


def compress_audio_chunk(chunk, target_bitrate="64k"):
    """Compress audio chunk to reduce file size"""
    try:
        # Export as MP3 with lower bitrate for smaller file size
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        chunk.export(temp_file.name, format="mp3", bitrate=target_bitrate)

        # Check file size
        file_size = os.path.getsize(temp_file.name)
        logger.info(f"Compressed chunk size: {file_size} bytes")

        if file_size > MAX_FILE_SIZE_BYTES:
            # Try even lower bitrate
            os.unlink(temp_file.name)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            chunk.export(temp_file.name, format="mp3", bitrate="32k")
            file_size = os.path.getsize(temp_file.name)
            logger.info(f"Further compressed chunk size: {file_size} bytes")

        return temp_file.name, file_size
    except Exception as e:
        logger.error(f"Error compressing audio chunk: {e}")
        # Fallback to WAV
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        chunk.export(temp_file.name, format="wav")
        return temp_file.name, os.path.getsize(temp_file.name)


@app.on_event("startup")
async def startup_event():
    logger.info("=" * 50)
    logger.info("Transcripter application starting up")
    logger.info(f"Python version: {os.sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"OpenAI API key configured: {'Yes' if api_key else 'No'}")
    logger.info("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Transcripter application shutting down")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    logger.info(f"Index page requested from {request.client.host}")
    available_transcripts = get_available_transcripts()
    logger.info(f"Found {len(available_transcripts)} available transcripts")
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "available_transcripts": available_transcripts},
    )


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0",
        "openai_configured": bool(api_key),
    }


@app.get("/progress/{session_id}")
async def get_progress(session_id: str):
    """Get current progress for a transcription session"""
    logger.info(f"Progress requested for session: {session_id}")

    if session_id in transcription_progress:
        return transcription_progress[session_id]
    else:
        return {
            "status": "waiting",
            "progress": 0,
            "message": "Waiting for transcription to start...",
            "current_chunk": 0,
            "total_chunks": 0,
        }


@app.get("/download/{filename}")
async def download_transcript(filename: str):
    """Download transcript file"""
    logger.info(f"Download requested for file: {filename}")

    # Security check - only allow .txt files with transcript_ prefix
    if not filename.startswith("transcript_") or not filename.endswith(".txt"):
        logger.error(f"Invalid filename for download: {filename}")
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Look for the file in transcripts directory
    file_path = TRANSCRIPTS_DIR / filename

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")

    logger.info(f"Serving file: {file_path}")

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.post("/transcribe", response_class=HTMLResponse)
async def transcribe(request: Request, file: UploadFile, session_id: str = None):
    # Generate session ID if not provided
    if not session_id:
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    start_time = datetime.now()
    logger.info(
        f"Transcription request started from {request.client.host} (Session: {session_id})"
    )
    logger.info(
        f"File received: {file.filename}, Content-Type: {file.content_type}, Size: {file.size} bytes"
    )

    # Initialize progress tracking
    transcription_progress[session_id] = {
        "status": "starting",
        "progress": 0,
        "message": "Starting transcription...",
        "current_chunk": 0,
        "total_chunks": 0,
    }

    try:
        # Validate file
        if not file.filename:
            logger.error("No filename provided")
            transcription_progress[session_id] = {
                "status": "error",
                "progress": 0,
                "message": "No file provided",
            }
            raise HTTPException(status_code=400, detail="No file provided")

        if not file.content_type or not file.content_type.startswith("audio/"):
            logger.error(f"Invalid file type: {file.content_type}")
            transcription_progress[session_id] = {
                "status": "error",
                "progress": 0,
                "message": "File must be an audio file",
            }
            raise HTTPException(status_code=400, detail="File must be an audio file")

        # Update progress
        transcription_progress[session_id] = {
            "status": "uploading",
            "progress": 10,
            "message": "Saving uploaded file...",
            "current_chunk": 0,
            "total_chunks": 0,
        }

        # Save uploaded file to a temp location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        logger.info(f"Saving uploaded file to temporary location: {temp_file.name}")

        async with aiofiles.open(temp_file.name, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)

        logger.info(f"File saved successfully, size: {len(content)} bytes")

        # Update progress
        transcription_progress[session_id] = {
            "status": "processing",
            "progress": 20,
            "message": "Loading and processing audio file...",
            "current_chunk": 0,
            "total_chunks": 0,
        }

        # Load audio with pydub
        logger.info("Loading audio file with pydub")
        audio = AudioSegment.from_mp3(temp_file.name)
        original_duration = len(audio) / 1000  # Convert to seconds
        logger.info(
            f"Audio loaded: {original_duration:.2f} seconds, {audio.channels} channels, {audio.frame_rate} Hz"
        )

        # Convert to mono and 16kHz for better transcription
        audio = audio.set_channels(1).set_frame_rate(16000)
        logger.info("Audio converted to mono, 16kHz")

        # Calculate optimal chunk size based on audio duration
        optimal_chunk_ms = get_optimal_chunk_size(len(audio))
        num_chunks = math.ceil(len(audio) / optimal_chunk_ms)
        logger.info(
            f"Audio will be split into {num_chunks} chunks of max {optimal_chunk_ms / 1000:.1f} seconds each"
        )

        # Update progress with chunk information
        transcription_progress[session_id] = {
            "status": "transcribing",
            "progress": 30,
            "message": f"Starting transcription of {num_chunks} chunks...",
            "current_chunk": 0,
            "total_chunks": num_chunks,
        }

        full_transcript = ""

        for i in range(num_chunks):
            start = i * optimal_chunk_ms
            end = min((i + 1) * optimal_chunk_ms, len(audio))
            if start >= len(audio):
                break

            chunk_duration = (end - start) / 1000

            # Update progress for current chunk
            progress_percent = 30 + (i / num_chunks) * 60  # 30% to 90%
            transcription_progress[session_id] = {
                "status": "transcribing",
                "progress": int(progress_percent),
                "message": f"Transcribing chunk {i + 1}/{num_chunks} ({chunk_duration:.2f}s)...",
                "current_chunk": i + 1,
                "total_chunks": num_chunks,
            }

            logger.info(
                f"Processing chunk {i + 1}/{num_chunks} ({chunk_duration:.2f} seconds)"
            )

            chunk = audio[start:end]

            # Compress chunk to reduce file size
            chunk_file, file_size = compress_audio_chunk(chunk)
            logger.info(f"Chunk {i + 1} compressed to {file_size} bytes")

            # Validate file size before sending to OpenAI
            if file_size > MAX_FILE_SIZE_BYTES:
                logger.error(f"Chunk {i + 1} still too large: {file_size} bytes")
                transcription_progress[session_id] = {
                    "status": "error",
                    "progress": int(progress_percent),
                    "message": f"Chunk {i + 1} too large even after compression: {file_size} bytes",
                    "current_chunk": i + 1,
                    "total_chunks": num_chunks,
                }
                os.unlink(chunk_file)
                raise HTTPException(
                    status_code=500, detail=f"Chunk {i + 1} too large for OpenAI API"
                )

            try:
                with open(chunk_file, "rb") as f:
                    logger.info(
                        f"Sending chunk {i + 1} to OpenAI for transcription ({file_size} bytes)"
                    )
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",  # Using whisper-1 for better compatibility
                        file=f,
                    )
                    chunk_text = transcript.text.strip()
                    full_transcript += chunk_text + "\n"
                    logger.info(
                        f"Chunk {i + 1} transcribed successfully ({len(chunk_text)} characters)"
                    )
            except Exception as e:
                logger.error(f"Error transcribing chunk {i + 1}: {str(e)}")
                transcription_progress[session_id] = {
                    "status": "error",
                    "progress": int(progress_percent),
                    "message": f"Error transcribing chunk {i + 1}: {str(e)}",
                    "current_chunk": i + 1,
                    "total_chunks": num_chunks,
                }
                os.unlink(chunk_file)
                raise HTTPException(
                    status_code=500, detail=f"Transcription failed for chunk {i + 1}"
                )

            os.unlink(chunk_file)  # clean up chunk
            logger.info(f"Chunk {i + 1} temporary file cleaned up")

        os.unlink(temp_file.name)  # clean up original mp3
        logger.info("Original temporary file cleaned up")

        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Transcription completed successfully in {total_time:.2f} seconds")
        logger.info(f"Total transcript length: {len(full_transcript)} characters")

        # Update progress for saving
        transcription_progress[session_id] = {
            "status": "saving",
            "progress": 90,
            "message": "Saving transcript to storage...",
            "current_chunk": num_chunks,
            "total_chunks": num_chunks,
        }

        # Save transcript to persistent storage
        transcript_filename = (
            f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        transcript_file_path = TRANSCRIPTS_DIR / transcript_filename

        with open(transcript_file_path, "w", encoding="utf-8") as f:
            f.write(f"Transcription of: {file.filename}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {original_duration:.2f} seconds\n")
            f.write(f"Processing time: {total_time:.2f} seconds\n")
            f.write("-" * 50 + "\n\n")
            f.write(full_transcript)

        logger.info(f"Transcript saved to: {transcript_file_path}")

        # Add to metadata
        add_transcript_to_metadata(
            transcript_filename,
            file.filename,
            f"{original_duration:.2f}",
            f"{total_time:.2f}",
            len(full_transcript),
        )

        # Final progress update
        transcription_progress[session_id] = {
            "status": "completed",
            "progress": 100,
            "message": f"Transcription completed! {len(full_transcript)} characters transcribed.",
            "current_chunk": num_chunks,
            "total_chunks": num_chunks,
            "transcript_filename": transcript_filename,
            "processing_time": f"{total_time:.2f}",
            "audio_duration": f"{original_duration:.2f}",
        }

        # Get updated list of available transcripts
        available_transcripts = get_available_transcripts()

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "transcript": full_transcript,
                "transcript_filename": transcript_filename,
                "transcript_file_path": str(transcript_file_path),
                "original_filename": file.filename,
                "processing_time": f"{total_time:.2f}",
                "audio_duration": f"{original_duration:.2f}",
                "available_transcripts": available_transcripts,
                "session_id": session_id,
            },
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error during transcription: {str(e)}", exc_info=True)
        # Clean up temp files in case of error
        try:
            if "temp_file" in locals():
                os.unlink(temp_file.name)
        except OSError:
            pass
        raise HTTPException(
            status_code=500, detail="Internal server error during transcription"
        )
