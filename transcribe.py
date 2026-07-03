import argparse
import os
import subprocess
import sys
import tempfile
from itertools import islice
from pathlib import Path

import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

SAMPLE_RATE = 16_000
BATCH_SIZE = 16
MAX_SEGMENT_S = 25

WHISPER_MODEL = "openai/whisper-large-v3-turbo"
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def to_mono_wav(path: str) -> tuple[str, torch.Tensor]:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    try:
        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0, keepdim=True) if wav.shape[0] > 1 else wav
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE) if sr != SAMPLE_RATE else wav
        torchaudio.save(tmp.name, wav, SAMPLE_RATE)
        return tmp.name, wav
    except Exception:
        subprocess.run(
            ["ffmpeg", "-y", "-nostdin", "-i", path, "-ar", str(SAMPLE_RATE), "-ac", "1", tmp.name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        wav, _ = torchaudio.load(tmp.name)
        return tmp.name, wav

def clean(text: str) -> str:
    text = text.strip()
    if len(text) < 2:
        return ""

    words = text.split()
    repeated_word = len(words) >= 3 and len(set(words)) == 1
    too_much_punctuation = text.count("!") / len(text) > 0.2
    too_much_non_ascii = sum(not c.isascii() for c in text) / len(text) > 0.05

    return "" if repeated_word or too_much_punctuation or too_much_non_ascii else text

def merge_segments(segments, max_gap=0.4, max_len=MAX_SEGMENT_S, min_len=0.2):
    segments = [(s, e, spk) for s, e, spk in segments if e - s >= min_len]
    if not segments:
        return []

    merged = [segments[0]]

    for start, end, speaker in segments[1:]:
        prev_start, prev_end, prev_speaker = merged[-1]
        can_merge = (
            speaker == prev_speaker
            and start - prev_end <= max_gap
            and end - prev_start <= max_len
        )

        if can_merge:
            merged[-1] = (prev_start, end, speaker)
        else:
            merged.append((start, end, speaker))

    return merged

class WhisperTranscriber:
    def __init__(self, model_name: str = WHISPER_MODEL):
        self.dev = device()
        self.dtype = torch.float32 if self.dev.type == "cpu" else torch.float16
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.dev)

        self.model.config.forced_decoder_ids = None
        self.model.generation_config.forced_decoder_ids = None

    def __call__(self, chunks: list) -> list[str]:
        inputs = self.processor(chunks, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        features = inputs.input_features.to(self.dev, dtype=self.dtype)
        mask = inputs.get("attention_mask")
        mask = mask.to(self.dev) if mask is not None else None

        with torch.no_grad():
            ids = self.model.generate(
                features,
                attention_mask=mask,
                language="en",
                task="transcribe",
                temperature=0.0,
                no_repeat_ngram_size=4,
                repetition_penalty=1.2,
                condition_on_prev_tokens=False,
            )

        return [clean(t) for t in self.processor.batch_decode(ids, skip_special_tokens=True)]

class Diarizer:
    def __init__(self, hf_token: str):
        self.pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=hf_token)
        self.pipeline.to(device())

    def __call__(self, path: str, speakers=2):
        with ProgressHook() as hook:
            return self.pipeline(path, hook=hook, min_speakers=speakers, max_speakers=speakers)

def segment_audio(audio, segments):
    for start, end, speaker in segments:
        sample_start = int(start * SAMPLE_RATE)
        sample_end = int(end * SAMPLE_RATE)
        yield audio[sample_start:sample_end], sample_start, sample_end, start, end, speaker

def batched(items, size):
    iterator = iter(items)
    index = 0
    while batch := list(islice(iterator, size)):
        yield index, batch
        index += len(batch)

def transcribe(path: str, hf_token: str, model_name: str = WHISPER_MODEL):
    print("Preparing audio...")
    wav_path, waveform = to_mono_wav(path)

    try:
        print("Diarizing...")
        diarization = Diarizer(hf_token)(wav_path)
        segments = [
            (turn.start, turn.end, speaker)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
        segments = merge_segments(segments)

        print(f"{len(segments)} chunks to transcribe.")
        audio = waveform.squeeze(0).numpy()

        print(f"Loading Whisper model: {model_name}")
        whisper = WhisperTranscriber(model_name)
        results = []

        for offset, batch in batched(segment_audio(audio, segments), BATCH_SIZE):
            samples = [sample for sample, _, _, _, _, _ in batch]
            texts = whisper(samples)

            for (_, _, _, start, end, speaker), text in zip(batch, texts):
                if text:
                    results.append((start, end, speaker, text))

            print(f"Processed {offset + len(batch)}/{len(segments)} chunks.")

        print("\n--- Transcript ---\n")
        for start, end, speaker, text in sorted(results):
            print(f"[{start:.1f}s -> {end:.1f}s] {speaker}: {text}")

    finally:
        Path(wav_path).unlink(missing_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("hf_token", nargs="?")
    parser.add_argument("--model", default=WHISPER_MODEL)
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        print("Error: pass a Hugging Face token or set HF_TOKEN.")
        sys.exit(1)

    transcribe(args.audio, token, args.model)

if __name__ == "__main__":
    main()
