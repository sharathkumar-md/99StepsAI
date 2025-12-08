"""
Whisper ASR (Automatic Speech Recognition) Module
This module uses OpenAI's Whisper model to transcribe audio files to text.
"""

import os
import torch
import whisper
import logging
from pathlib import Path
import json
from typing import Optional, Dict, Any
import warnings

warnings.filterwarnings("ignore")

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('logs/chatbot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WhisperASR:
    """Whisper-based Automatic Speech Recognition"""

    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Initialize Whisper ASR

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to use (cuda/cpu). If None, auto-detect
        """
        self.model_size = model_size

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading Whisper {model_size} model on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)
        logger.info("Model loaded successfully!")

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe audio file to text

        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en' for English). If None, auto-detect
            task: 'transcribe' or 'translate' (to English)
            verbose: Whether to print transcription progress

        Returns:
            Dictionary containing transcription results
        """
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing: {audio_path}")

        # Transcribe
        result = self.model.transcribe(
            audio_path,
            language=language,
            task=task,
            verbose=verbose
        )

        logger.info(f"Transcription complete. Detected language: {result.get('language', 'unknown')}")
        return result

    def transcribe_with_timestamps(
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio with word-level timestamps

        Args:
            audio_path: Path to audio file
            language: Language code. If None, auto-detect

        Returns:
            Dictionary with transcription and segment-level timestamps
        """
        result = self.transcribe(audio_path, language=language)

        # Format output with timestamps
        output = {
            "text": result["text"],
            "language": result["language"],
            "segments": []
        }

        for segment in result["segments"]:
            output["segments"].append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip()
            })

        logger.debug(f"Processed {len(output['segments'])} segments")
        return output

    def save_transcription(
        self,
        transcription: Dict[str, Any],
        output_path: str,
        format: str = "json"
    ):
        """
        Save transcription to file

        Args:
            transcription: Transcription result dictionary
            output_path: Output file path
            format: Output format ('json' or 'txt')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if format == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(transcription, f, indent=2, ensure_ascii=False)
                logger.info(f"Transcription saved to: {output_path}")

            elif format == "txt":
                with open(output_path, 'w', encoding='utf-8') as f:
                    if "segments" in transcription:
                        for seg in transcription["segments"]:
                            f.write(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}\n")
                    else:
                        f.write(transcription["text"])
                logger.info(f"Transcription saved to: {output_path}")
            else:
                logger.error(f"Unsupported format: {format}")
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            logger.error(f"Error saving transcription: {e}")
            raise


def main():
    """Example usage of WhisperASR"""

    # Example audio file path (you can change this)
    audio_path = "sample_audio.wav"

    # Check if sample audio exists
    if not os.path.exists(audio_path):
        logger.warning(f"No audio file found at '{audio_path}'")
        logger.info("Please provide an audio file to transcribe.")
        logger.info("Usage:")
        logger.info("  asr = WhisperASR(model_size='base')")
        logger.info("  result = asr.transcribe('your_audio.wav')")
        logger.info("  asr.save_transcription(result, 'output.json')")
        return

    # Initialize ASR
    asr = WhisperASR(model_size="base")

    # Transcribe with timestamps
    logger.info("="*60)
    logger.info("WHISPER ASR - TRANSCRIPTION")
    logger.info("="*60)

    result = asr.transcribe_with_timestamps(audio_path)

    # Log results
    logger.info(f"Detected Language: {result['language']}")
    logger.info(f"Full Transcription: {result['text']}")

    logger.info("Segmented Transcription:")
    for seg in result['segments']:
        logger.info(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")

    # Save results
    asr.save_transcription(result, "transcription.json", format="json")
    asr.save_transcription(result, "transcription.txt", format="txt")

    logger.info("="*60)
    logger.info("Transcription Complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
