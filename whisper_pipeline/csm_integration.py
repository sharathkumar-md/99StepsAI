"""
CSM Integration Module
Handles text-to-speech conversion using CSM (Conversational Speech Model)
"""

# Suppress ALL warnings BEFORE any imports
import os
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import warnings
warnings.filterwarnings('ignore')

import sys
import logging
import torch
import torchaudio
from pathlib import Path
import tempfile

# Add parent directory to path to import CSM
sys.path.insert(0, str(Path(__file__).parent.parent / 'csm'))

from generator import load_csm_1b, Segment

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


class CSMConverter:
    """Convert text to audio using CSM"""

    def __init__(self, device: str = None):
        """
        Initialize CSM converter

        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        # Comprehensive suppression of torch warnings
        import warnings
        import logging as builtin_logging

        # Disable Triton compilation
        os.environ["NO_TORCH_COMPILE"] = "1"
        os.environ["TORCHDYNAMO_DISABLE"] = "1"

        # Suppress all warnings from torch, triton, and dynamo
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', module='torch')
        warnings.filterwarnings('ignore', module='triton')
        warnings.filterwarnings('ignore', module='torch._dynamo')

        # Disable triton warnings at logging level
        builtin_logging.getLogger('triton').setLevel(builtin_logging.ERROR)

        # Suppress torch dynamo completely
        import torch
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.verbose = False

        # Disable compilation
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device
        logger.info(f"Initializing CSM on device: {device}")

        # Load CSM model
        try:
            self.generator = load_csm_1b(device=device)
            self.sample_rate = self.generator.sample_rate
            logger.info(f"CSM loaded successfully (sample rate: {self.sample_rate}Hz)")
        except Exception as e:
            logger.error(f"Failed to load CSM: {e}")
            raise

    def text_to_audio(
        self,
        text: str,
        speaker: int = 0,
        max_audio_length_ms: float = 30000,
        output_path: str = None
    ) -> str:
        """
        Convert text to audio using CSM

        Args:
            text: Text to convert to speech
            speaker: Speaker ID (0 or 1)
            max_audio_length_ms: Maximum audio length in milliseconds
            output_path: Optional path to save audio file

        Returns:
            Path to generated audio file
        """
        if not text or not text.strip():
            logger.warning("Empty text input, cannot generate audio")
            return None

        logger.info(f"Converting text to audio: '{text[:50]}...'")

        try:
            # Generate audio with CSM
            audio_tensor = self.generator.generate(
                text=text,
                speaker=speaker,
                context=[],
                max_audio_length_ms=max_audio_length_ms
            )

            # Create temp file if no output path specified
            if output_path is None:
                temp_file = tempfile.NamedTemporaryFile(
                    suffix='.wav',
                    delete=False
                )
                output_path = temp_file.name
                temp_file.close()

            # Save audio
            torchaudio.save(
                output_path,
                audio_tensor.unsqueeze(0).cpu(),
                self.sample_rate
            )

            logger.info(f"Audio generated successfully: {output_path}")
            logger.debug(f"Audio duration: {len(audio_tensor) / self.sample_rate:.2f}s")

            return output_path

        except Exception as e:
            logger.error(f"Failed to generate audio: {e}")
            raise

    def batch_text_to_audio(
        self,
        texts: list,
        speaker: int = 0,
        output_dir: str = None
    ) -> list:
        """
        Convert multiple texts to audio

        Args:
            texts: List of texts to convert
            speaker: Speaker ID
            output_dir: Directory to save audio files

        Returns:
            List of audio file paths
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        audio_paths = []

        for i, text in enumerate(texts, 1):
            logger.info(f"Processing text {i}/{len(texts)}")

            if output_dir:
                output_path = os.path.join(output_dir, f"audio_{i:03d}.wav")
            else:
                output_path = None

            audio_path = self.text_to_audio(text, speaker, output_path=output_path)
            audio_paths.append(audio_path)

        logger.info(f"Generated {len(audio_paths)} audio files")
        return audio_paths


def main():
    """Test CSM integration"""
    logger.info("="*60)
    logger.info("CSM INTEGRATION TEST")
    logger.info("="*60)

    # Initialize CSM
    logger.info("\nInitializing CSM...")
    csm = CSMConverter()

    # Test text-to-audio
    test_text = "Hello, this is a test of the CSM text to speech system."
    logger.info(f"\nTest: Converting text to audio")
    logger.info(f"Text: {test_text}")

    audio_path = csm.text_to_audio(test_text, output_path="test_csm_output.wav")

    logger.info(f"\nAudio saved to: {audio_path}")
    logger.info("✓ CSM integration test complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
