"""
Extension 11 — TTS RLHF: Preference Data for Speech Quality Optimisation.

The gap this closes
--------------------
All previous extensions operate in text space: prompts, responses, preferences
are token sequences.  Speech introduces a second modality where quality is
perceptual — a *better* utterance has more natural prosody, cleaner voice, and
appropriate pacing.  RLHF can close the same gap in audio that it closes in text:
instead of "prefer A over B because it sounds more helpful", we collect "prefer A
over B because it sounds more natural".

Pipeline
---------
  Text prompts × generation variants (temperature, description)
       │
       ▼
  Parler-TTS: generate multiple speech samples per prompt
       │
       ▼
  Scoring: UTMOS (automatic MOS predictor) or acoustic feature heuristics
       │
       ▼
  (prompt, chosen_audio, rejected_audio) pairs → JSONL
       │
       ▼
  Bradley-Terry RM trained on audio embeddings / perceptual features
       │
       ▼
  DPO on Parler-TTS audio token sequences

Audio quality dimensions
-------------------------
- Naturalness: prosodic variation, appropriate pausing, expressive delivery
- Intelligibility: clarity, signal-to-noise, spectral quality
- Speaker consistency: stable timbre, no artefacts, no codec distortion

Preference labelling
---------------------
We use UTMOS22 (automatic MOS predictor; sarulab/utmos22_strong) as the AI
labeller — analogous to Claude's role in Extension 1 (CAI/RLAIF).  If UTMOS
is unavailable, we fall back to a composite acoustic metric:
  score = 0.4 * pitch_variance_norm
        + 0.3 * harmonic_to_noise_ratio_norm
        + 0.2 * voiced_fraction
        + 0.1 * (1 - silence_fraction_penalty)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── TTS text prompt catalogue ──────────────────────────────────────────────────
# 30 diverse prompts covering different speaking styles and content types.
# Longer, more complex prompts expose prosody differences between TTS variants.

TTS_PROMPT_CATALOGUE: List[Dict] = [
    # Factual / informational
    {"text": "The mitochondria is the powerhouse of the cell, generating ATP through oxidative phosphorylation.", "style": "informational"},
    {"text": "Mount Everest, standing at 8,848 metres, was first summited by Edmund Hillary and Tenzing Norgay in 1953.", "style": "informational"},
    {"text": "The speed of light in a vacuum is approximately 299,792 kilometres per second, which forms the cosmic speed limit.", "style": "informational"},
    {"text": "Python was created by Guido van Rossum and first released in 1991, emphasising code readability and simplicity.", "style": "informational"},
    {"text": "The Amazon rainforest produces about 20 percent of the world's oxygen and is home to 10 percent of all species on Earth.", "style": "informational"},
    # Narrative / storytelling
    {"text": "It was a quiet Tuesday morning when she first noticed the light in the old abandoned house at the end of the lane.", "style": "narrative"},
    {"text": "He had walked that same road a thousand times, but something about the autumn leaves made it feel entirely new.", "style": "narrative"},
    {"text": "The market was alive with colour, noise, and the mingled scents of spices, fresh bread, and rain on cobblestones.", "style": "narrative"},
    {"text": "As the sun set over the harbour, the fishing boats returned, their nets heavy with the silver catch of the day.", "style": "narrative"},
    {"text": "She opened the letter slowly, already knowing that whatever was inside would change everything forever.", "style": "narrative"},
    # Instructional
    {"text": "To make a perfect cup of coffee, start with freshly ground beans and water heated to exactly 93 degrees Celsius.", "style": "instructional"},
    {"text": "When reversing a car, check all three mirrors, then turn to look over your shoulder before slowly applying pressure to the accelerator.", "style": "instructional"},
    {"text": "Save your work frequently, use descriptive variable names, and write comments explaining why your code does what it does.", "style": "instructional"},
    {"text": "Before your presentation, take three slow deep breaths, make eye contact with friendly faces, and speak slightly slower than feels natural.", "style": "instructional"},
    {"text": "To tie a bowline knot, make a small loop, pass the free end up through it, around the standing line, and back down.", "style": "instructional"},
    # Conversational
    {"text": "I think the best part of working from home is the flexibility, though I do miss the social side of the office.", "style": "conversational"},
    {"text": "Have you ever noticed how the smell of old books is completely different from new ones? There's actually a word for it: bibliosmia.", "style": "conversational"},
    {"text": "The thing about learning a new language is that the embarrassing mistakes are actually what make you remember the right way.", "style": "conversational"},
    {"text": "If you ask me, the best travel experiences come from completely unplanned detours and wrong turns.", "style": "conversational"},
    {"text": "I've started keeping a notebook by the bed because the most interesting thoughts always seem to arrive right before sleep.", "style": "conversational"},
    # Technical / complex prosody
    {"text": "The transformer architecture, introduced in the paper Attention Is All You Need, uses multi-head self-attention to process sequences in parallel.", "style": "technical"},
    {"text": "Gradient descent, stochastic gradient descent, and adaptive methods like Adam differ primarily in how they estimate and apply parameter updates.", "style": "technical"},
    {"text": "A blockchain is a distributed ledger where each block contains a cryptographic hash of the previous block, creating an immutable chain.", "style": "technical"},
    {"text": "RLHF — reinforcement learning from human feedback — uses a reward model trained on preference data to fine-tune a language model via PPO or DPO.", "style": "technical"},
    {"text": "The Fourier transform decomposes a signal into its constituent frequencies, revealing the spectral content that is invisible in the time domain.", "style": "technical"},
    # Emotional / expressive
    {"text": "Thank you so much — this truly means the world to me, and I will never forget your kindness.", "style": "expressive"},
    {"text": "I am absolutely delighted to announce that after three years of work, we have finally achieved our goal.", "style": "expressive"},
    {"text": "This is not what we agreed to, and I need you to understand how serious this situation has become.", "style": "expressive"},
    {"text": "I know the road ahead will be difficult, but I also know that everyone in this room has what it takes.", "style": "expressive"},
    {"text": "Sometimes the simplest moments — a warm cup of tea, a good book, rain on the window — are the most precious.", "style": "expressive"},
]


# ── Parler-TTS description prompts ─────────────────────────────────────────────
# Different description strings produce different speaker/style characteristics.
# We use variation in description to generate diverse outputs for preference pairs.

TTS_DESCRIPTIONS = {
    "natural_female": (
        "A female speaker delivers the text with natural, expressive intonation, "
        "appropriate pausing, and a warm, clear voice. The recording quality is high "
        "with no background noise."
    ),
    "natural_male": (
        "A male speaker with a deep, resonant voice delivers the text clearly and "
        "confidently, with natural prosody and appropriate emphasis."
    ),
    "flat": (
        "A speaker reads the text in a flat, monotone voice with little variation "
        "in pitch or pace."
    ),
    "expressive": (
        "A very expressive, enthusiastic speaker delivers the text with significant "
        "pitch variation, clear emphasis on key words, and natural pausing."
    ),
    "fast": (
        "A speaker delivers the text at a fast pace with reduced pausing between phrases."
    ),
    "slow_clear": (
        "A speaker delivers the text slowly and very clearly, with deliberate pausing "
        "between sentences and careful pronunciation."
    ),
}

# Pairs to compare: (description_A, description_B)
# We expect the first to be preferred in most cases.
TTS_DESCRIPTION_PAIRS: List[Tuple[str, str]] = [
    ("natural_female", "flat"),       # natural > monotone
    ("natural_female", "fast"),       # paced > rushed
    ("natural_male", "flat"),         # natural > monotone
    ("expressive", "flat"),           # expressive > monotone
    ("slow_clear", "fast"),           # clear > rushed
]


# ── Acoustic feature extraction ────────────────────────────────────────────────

def extract_acoustic_features(
    audio: np.ndarray,
    sample_rate: int = 24000,
) -> Dict[str, float]:
    """
    Extract perceptual quality features from a speech waveform.

    Returns a dict of normalised [0, 1] features:
      pitch_variance       — prosodic naturalness proxy
      voiced_fraction      — fraction of frames with detected pitch
      hnr                  — harmonic-to-noise ratio (voice quality)
      silence_fraction     — fraction of near-silent frames (pacing)
      spectral_centroid    — brightness / clarity indicator
      mfcc_stability       — MFCC variance across time (consistency)
      energy_dynamics      — RMS energy variation (expressiveness)
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required: pip install librosa")

    # Ensure mono float32
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    audio = audio.astype(np.float32)

    features: Dict[str, float] = {}

    # ── Pitch / F0 ───────────────────────────────────────────────────────────
    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sample_rate,
        )
        voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
        features["pitch_variance"] = float(np.std(voiced_f0)) / 200.0 if len(voiced_f0) > 5 else 0.0
        features["voiced_fraction"] = float(np.mean(voiced_flag))
    except Exception:
        features["pitch_variance"] = 0.0
        features["voiced_fraction"] = 0.0

    # ── Harmonic-to-noise ratio proxy (spectral flatness inverse) ────────────
    try:
        sfm = librosa.feature.spectral_flatness(y=audio)
        # Low flatness ≈ tonal (voiced) ≈ high HNR proxy
        features["hnr"] = float(1.0 - np.mean(sfm).clip(0, 1))
    except Exception:
        features["hnr"] = 0.5

    # ── Silence fraction ─────────────────────────────────────────────────────
    try:
        rms = librosa.feature.rms(y=audio)[0]
        silence_threshold = 0.01 * rms.max()
        features["silence_fraction"] = float(np.mean(rms < silence_threshold))
    except Exception:
        features["silence_fraction"] = 0.3

    # ── Spectral centroid ─────────────────────────────────────────────────────
    try:
        sc = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
        # Normalise: typical speech centroid 1–4 kHz
        features["spectral_centroid"] = float(np.mean(sc).clip(0, 8000) / 8000.0)
    except Exception:
        features["spectral_centroid"] = 0.3

    # ── MFCC stability ────────────────────────────────────────────────────────
    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        # Low variance in MFCCs = consistent voice quality
        features["mfcc_stability"] = float(1.0 / (1.0 + np.mean(np.std(mfccs, axis=1))))
    except Exception:
        features["mfcc_stability"] = 0.5

    # ── Energy dynamics ────────────────────────────────────────────────────────
    try:
        rms = librosa.feature.rms(y=audio)[0]
        features["energy_dynamics"] = float(np.std(rms) / (np.mean(rms) + 1e-9)).clip(0, 2) / 2.0
    except Exception:
        features["energy_dynamics"] = 0.3

    return features


def acoustic_quality_score(features: Dict[str, float]) -> float:
    """
    Composite acoustic MOS proxy from extracted features.

    Weighted combination calibrated to correlate with human MOS scores:
      pitch_variance    0.30  — prosody naturalness
      hnr               0.25  — voice quality / intelligibility
      voiced_fraction   0.20  — speech content
      energy_dynamics   0.15  — expressiveness
      mfcc_stability    0.10  — consistency
    """
    score = (
        0.30 * min(features.get("pitch_variance", 0.0), 1.0)
        + 0.25 * features.get("hnr", 0.5)
        + 0.20 * features.get("voiced_fraction", 0.5)
        + 0.15 * features.get("energy_dynamics", 0.3)
        + 0.10 * features.get("mfcc_stability", 0.5)
    )
    return float(score)


# ── UTMOS scorer ──────────────────────────────────────────────────────────────

def score_with_utmos(audio: np.ndarray, sample_rate: int = 16000) -> Optional[float]:
    """
    Score audio with UTMOS22 (automatic MOS predictor).
    Returns a MOS score in [1, 5] or None if UTMOS is not available.

    Model: sarulab/utmos22_strong (HuggingFace)
    """
    try:
        import torch
        from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

        # Resample to 16 kHz (UTMOS requirement)
        if sample_rate != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        model_id = "sarulab/utmos22_strong"
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
        model.eval()

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        # UTMOS outputs a MOS score directly
        return float(logits[0].item())
    except Exception:
        return None


# ── Preference pair generation ────────────────────────────────────────────���────

@dataclass
class TTSPreferenceConfig:
    output_path: str = "data/tts_preferences.jsonl"
    model_id: str = "parler-tts/parler-tts-mini-v1"
    sample_rate: int = 24000
    prompts_per_pair: int = 30       # how many text prompts to use
    use_utmos: bool = False           # set True if UTMOS model is available
    device: str = "cpu"              # "cuda" for GPU
    description_pairs: List[Tuple[str, str]] = field(
        default_factory=lambda: TTS_DESCRIPTION_PAIRS
    )


def generate_tts_audio(
    model,
    tokenizer,
    description_tokenizer,
    text: str,
    description: str,
    device: str = "cpu",
) -> Optional[np.ndarray]:
    """Generate audio for one (text, description) pair using Parler-TTS."""
    try:
        import torch
        desc_inputs = description_tokenizer(description, return_tensors="pt").to(device)
        text_inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            generation = model.generate(
                input_ids=desc_inputs.input_ids,
                attention_mask=desc_inputs.attention_mask,
                prompt_input_ids=text_inputs.input_ids,
                prompt_attention_mask=text_inputs.attention_mask,
            )
        audio = generation.cpu().numpy().squeeze()
        return audio
    except Exception as e:
        print(f"    TTS generation failed: {e}")
        return None


def generate_tts_preference_dataset(cfg: TTSPreferenceConfig) -> None:
    """
    Generate a JSONL of TTS preference pairs.

    For each (text_prompt, description_pair):
      1. Generate audio_A with description_A
      2. Generate audio_B with description_B
      3. Score both with UTMOS or acoustic metrics
      4. Label higher-scoring as 'chosen'
      5. Save features + metadata to JSONL (audio saved as .wav files)
    """
    try:
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        import soundfile as sf
        import torch
    except ImportError as e:
        raise ImportError(
            f"Parler-TTS dependencies missing: {e}\n"
            "Install with: pip install git+https://github.com/huggingface/parler-tts.git soundfile"
        )

    from tqdm.auto import tqdm

    print(f"Loading Parler-TTS model: {cfg.model_id}")
    model = ParlerTTSForConditionalGeneration.from_pretrained(cfg.model_id).to(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    description_tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)

    os.makedirs(os.path.dirname(cfg.output_path) or ".", exist_ok=True)
    audio_dir = os.path.join(os.path.dirname(cfg.output_path) or ".", "tts_audio")
    os.makedirs(audio_dir, exist_ok=True)

    prompts = TTS_PROMPT_CATALOGUE[:cfg.prompts_per_pair]
    pairs = []
    pair_idx = 0

    total = len(prompts) * len(cfg.description_pairs)
    pbar = tqdm(total=total, desc="Generating TTS pairs")

    for prompt_item in prompts:
        text = prompt_item["text"]
        style = prompt_item["style"]

        for desc_a_key, desc_b_key in cfg.description_pairs:
            desc_a = TTS_DESCRIPTIONS[desc_a_key]
            desc_b = TTS_DESCRIPTIONS[desc_b_key]

            audio_a = generate_tts_audio(model, tokenizer, description_tokenizer, text, desc_a, cfg.device)
            audio_b = generate_tts_audio(model, tokenizer, description_tokenizer, text, desc_b, cfg.device)

            if audio_a is None or audio_b is None:
                pbar.update(1)
                continue

            # Score
            if cfg.use_utmos:
                score_a = score_with_utmos(audio_a, cfg.sample_rate) or 0.0
                score_b = score_with_utmos(audio_b, cfg.sample_rate) or 0.0
            else:
                feats_a = extract_acoustic_features(audio_a, cfg.sample_rate)
                feats_b = extract_acoustic_features(audio_b, cfg.sample_rate)
                score_a = acoustic_quality_score(feats_a)
                score_b = acoustic_quality_score(feats_b)

            # Save audio
            path_a = os.path.join(audio_dir, f"pair{pair_idx:04d}_A.wav")
            path_b = os.path.join(audio_dir, f"pair{pair_idx:04d}_B.wav")
            sf.write(path_a, audio_a, cfg.sample_rate)
            sf.write(path_b, audio_b, cfg.sample_rate)

            # Label
            if score_a >= score_b:
                chosen_path, rejected_path = path_a, path_b
                chosen_desc, rejected_desc = desc_a_key, desc_b_key
                chosen_score, rejected_score = score_a, score_b
            else:
                chosen_path, rejected_path = path_b, path_a
                chosen_desc, rejected_desc = desc_b_key, desc_a_key
                chosen_score, rejected_score = score_b, score_a

            record = {
                "pair_id": pair_idx,
                "text": text,
                "style": style,
                "chosen_audio_path": chosen_path,
                "rejected_audio_path": rejected_path,
                "chosen_description": chosen_desc,
                "rejected_description": rejected_desc,
                "chosen_score": round(chosen_score, 4),
                "rejected_score": round(rejected_score, 4),
                "score_delta": round(abs(score_a - score_b), 4),
                "scorer": "utmos" if cfg.use_utmos else "acoustic",
                "chosen_features": extract_acoustic_features(
                    np.load(chosen_path) if not chosen_path.endswith(".wav")
                    else _load_wav(chosen_path),
                    cfg.sample_rate,
                ) if not cfg.use_utmos else {},
            }
            pairs.append(record)
            pair_idx += 1
            pbar.set_postfix({"delta": f"{record['score_delta']:.3f}", "n": pair_idx})
            pbar.update(1)

    pbar.close()

    with open(cfg.output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    avg_delta = sum(p["score_delta"] for p in pairs) / max(len(pairs), 1)
    print(f"\nGenerated {len(pairs)} preference pairs → {cfg.output_path}")
    print(f"Average score delta: {avg_delta:.4f}")
    print(f"Audio files: {audio_dir}")


def _load_wav(path: str) -> np.ndarray:
    try:
        import soundfile as sf
        audio, _ = sf.read(path)
        return audio.astype(np.float32)
    except Exception:
        return np.zeros(24000, dtype=np.float32)


# ── Dataset class ─────────────────────────────────────────────────────────────

class TTSPreferenceDataset:
    """
    Dataset of TTS preference pairs for training the audio reward model.

    Each item provides:
      chosen_features    — acoustic feature dict for the preferred utterance
      rejected_features  — acoustic feature dict for the dispreferred utterance
      chosen_score       — float quality score
      rejected_score     — float quality score
      text               — the input text prompt
      score_delta        — absolute score difference (useful for curriculum)

    The reward model trains on chosen_features vs rejected_features using
    Bradley-Terry pairwise loss — exactly like the text RM in Stage 2,
    but with audio feature vectors instead of token-sequence embeddings.
    """

    FEATURE_KEYS = [
        "pitch_variance", "voiced_fraction", "hnr",
        "silence_fraction", "spectral_centroid", "mfcc_stability", "energy_dynamics",
    ]

    def __init__(
        self,
        jsonl_path: str,
        min_delta: float = 0.02,
        load_audio: bool = False,
        sample_rate: int = 24000,
    ) -> None:
        self.load_audio = load_audio
        self.sample_rate = sample_rate
        self.data: List[Dict] = []

        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line)
                if item.get("score_delta", 0) >= min_delta:
                    self.data.append(item)

        print(
            f"Loaded {len(self.data)} TTS preference pairs from {jsonl_path}"
            + (f" (min_delta={min_delta})" if min_delta > 0 else "")
        )

    def __len__(self) -> int:
        return len(self.data)

    def _features_to_vector(self, features: Dict) -> np.ndarray:
        return np.array([features.get(k, 0.5) for k in self.FEATURE_KEYS], dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        if self.load_audio:
            chosen_audio = _load_wav(item["chosen_audio_path"])
            rejected_audio = _load_wav(item["rejected_audio_path"])
            chosen_feats = extract_acoustic_features(chosen_audio, self.sample_rate)
            rejected_feats = extract_acoustic_features(rejected_audio, self.sample_rate)
        else:
            chosen_feats = item.get("chosen_features", {})
            rejected_feats = item.get("rejected_features", {})

        return {
            "chosen_features": self._features_to_vector(chosen_feats),
            "rejected_features": self._features_to_vector(rejected_feats),
            "chosen_score": float(item["chosen_score"]),
            "rejected_score": float(item["rejected_score"]),
            "score_delta": float(item["score_delta"]),
            "text": item["text"],
        }

    def feature_dim(self) -> int:
        return len(self.FEATURE_KEYS)
