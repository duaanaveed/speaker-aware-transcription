# speaker-aware-transcription
Speaker-aware speech transcription using Whisper &amp; Pyannote Audio

## Overview
This project combines speaker diarization and automatic speech recognition (ASR) to generate timestamped transcripts labelled by speaker.
The pipeline:
1. Converts audio to 16 kHz mono WAV.
2. Performs speaker diarization using Pyannote Audio.
3. Merges adjacent segments belonging to the same speaker.
4. Transcribes each segment using OpenAI Whisper.
5. Produces a timestamped transcript with speaker labels.

## Features
- Speaker diarization
- Automatic speech transcription
- Timestamped speaker-labelled output
- Automatic audio preprocessing
- Batch transcription for improved efficiency
- Supports CUDA, Apple Silicon (MPS), and CPU

## Requirements
- Python 3.10+
- PyTorch
- Torchaudio
- Transformers
- Pyannote Audio
- FFmpeg
- Hugging Face access token

## Installation
```bash
git clone https://github.com/<your-username>/speaker-aware-transcription.git
cd speaker-aware-transcription
pip install -r requirements.txt
```

## Usage=
```bash
python transcribe.py <audio_file> <hugging_face_token>
```
Or set your Hugging Face token as an environment variable:
```bash
export HF_TOKEN=<your_token>
python transcribe.py <audio_file>
```

## Example Output
```text
--- Transcript ---
[0.0s -> 3.8s] SPEAKER_00: Good morning everyone.
[3.9s -> 7.2s] SPEAKER_01: Thanks for joining today's meeting.
[7.3s -> 10.5s] SPEAKER_00: Let's get started.
```

## Pipeline
```
Audio
  │
  ▼
Audio preprocessing
  │
  ▼
Speaker diarization (Pyannote)
  │
  ▼
Segment merging
  │
  ▼
Speech transcription (Whisper)
  │
  ▼
Timestamped speaker-labelled transcript
```
## License

This project is licensed under the MIT License.
