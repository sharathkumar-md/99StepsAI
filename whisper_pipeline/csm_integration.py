"""
CSM Integration Module
Handles text-to-speech conversion using CSM (Conversational Speech Model)
OPTIMIZED FOR GPU (SageMaker g5.xlarge A10G)
"""

# Suppress ALL warnings BEFORE any imports
import os
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import warnings
warnings.filterwarnings('ignore')

import sys
import logging
import time
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

        # 🔥 Auto-detect device with explicit GPU check
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"✓ GPU Detected: {torch.cuda.get_device_name(0)}")
                logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                device = "cpu"
                logger.warning("⚠️  No GPU detected - CSM will be VERY slow on CPU!")

        self.device = device
        logger.info(f"CSM Device: {device}")
        
        # 🔥 Enable optimizations for GPU
        if device == "cuda":
            # Enable TF32 for faster matmul on Ampere GPUs (A10G)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 enabled for faster inference on A10G GPU")

        # Load CSM model with optimizations
        try:
            logger.info("Loading CSM model (sesame/csm-1b)...")
            logger.info(f"Target device: {device}")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            logger.info(f"PyTorch version: {torch.__version__}")
            logger.info(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
            
            load_start = time.time()
            
            # 🔥 CRITICAL: Load generator - it should handle device internally
            self.generator = load_csm_1b(device=device)
            self.sample_rate = self.generator.sample_rate
            
            # 🔥 Verify the model is actually on GPU
            if device == "cuda":
                logger.info("Verifying GPU placement...")
                
                # Check model device
                model_device = next(self.generator._model.parameters()).device
                logger.info(f"Model device: {model_device}")
                
                # Check model dtype
                model_dtype = next(self.generator._model.parameters()).dtype
                logger.info(f"Model dtype: {model_dtype}")
                
                # Check vocoder device
                if hasattr(self.generator, '_audio_tokenizer'):
                    vocoder_params = list(self.generator._audio_tokenizer.parameters())
                    if vocoder_params:
                        vocoder_device = vocoder_params[0].device
                        logger.info(f"Vocoder device: {vocoder_device}")
                
                # Force synchronization and check memory
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated(0) / 1e9
                reserved = torch.cuda.memory_reserved(0) / 1e9
                
                logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                
                if allocated < 0.5:
                    logger.error("❌ CRITICAL: Model not on GPU! Less than 0.5GB allocated!")
                    logger.error("The CSM generator.py load_csm_1b() function may have a bug")
                    logger.error(f"Model thinks it's on: {model_device}")
                    logger.error("CSM will run VERY slowly on CPU")
                elif str(model_device) == "cpu":
                    logger.error(f"❌ CRITICAL: Model is on CPU despite device={device}!")
                    logger.error("There's a bug in load_csm_1b() - it's not respecting device parameter")
                else:
                    logger.info(f"✅ Model successfully on GPU ({allocated:.2f}GB)")
            
            load_time = time.time() - load_start
            logger.info(f"✓ CSM loaded in {load_time:.2f}s (sample rate: {self.sample_rate}Hz)")
                
        except Exception as e:
            logger.error(f"Failed to load CSM: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
            # 🔥 CRITICAL: Measure generation time
            gen_start = time.time()
            
            # 🔥 Check GPU before generation
            if self.device == "cuda":
                before_mem = torch.cuda.memory_allocated(0) / 1e9
                logger.debug(f"GPU Memory before generation: {before_mem:.2f}GB")
            
            # Generate audio with CSM
            logger.debug("Generating audio tokens...")
            token_start = time.time()
            
            # 🔥 FORCE: Ensure generation uses GPU
            with torch.cuda.device(0) if self.device == "cuda" else torch.no_grad():
                audio_tensor = self.generator.generate(
                    text=text,
                    speaker=speaker,
                    context=[],
                    max_audio_length_ms=max_audio_length_ms
                )
            
            token_time = time.time() - token_start
            logger.debug(f"Token generation: {token_time:.2f}s")
            
            # 🔥 Check GPU after generation
            if self.device == "cuda":
                torch.cuda.synchronize()
                after_mem = torch.cuda.memory_allocated(0) / 1e9
                logger.debug(f"GPU Memory after generation: {after_mem:.2f}GB")
                if after_mem - before_mem < 0.1:
                    logger.warning("⚠️  GPU memory didn't increase during generation - may be using CPU!")

            # Create temp file if no output path specified
            if output_path is None:
                temp_file = tempfile.NamedTemporaryFile(
                    suffix='.wav',
                    delete=False
                )
                output_path = temp_file.name
                temp_file.close()

            # Save audio
            logger.debug("Saving audio file...")
            save_start = time.time()
            
            torchaudio.save(
                output_path,
                audio_tensor.unsqueeze(0).cpu(),
                self.sample_rate
            )
            
            save_time = time.time() - save_start
            total_time = time.time() - gen_start
            audio_duration = len(audio_tensor) / self.sample_rate

            logger.info(f"Audio generated successfully: {output_path}")
            logger.debug(f"Audio duration: {audio_duration:.2f}s")
            logger.debug(f"Performance - Generation: {token_time:.2f}s, Save: {save_time:.2f}s, Total: {total_time:.2f}s")
            
            # 🔥 GPU utilization check
            if self.device == "cuda":
                allocated = torch.cuda.memory_allocated(0) / 1e9
                logger.debug(f"GPU Memory: {allocated:.2f}GB in use")

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

    def __del__(self):
        """🔥 Cleanup GPU memory on deletion"""
        if hasattr(self, 'generator') and self.device == "cuda":
            try:
                del self.generator
                torch.cuda.empty_cache()
                logger.debug("CSM model cleaned up from GPU")
            except:
                pass


def main():
    """Test CSM integration with GPU diagnostics"""
    logger.info("="*60)
    logger.info("CSM INTEGRATION TEST (GPU Optimized)")
    logger.info("="*60)
    
    # GPU Diagnostics
    if torch.cuda.is_available():
        logger.info(f"\n🔍 GPU Diagnostics:")
        logger.info(f"  Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"  CUDA Version: {torch.version.cuda}")
        logger.info(f"  Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        logger.info(f"  Available VRAM: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9:.2f} GB")
    else:
        logger.warning("⚠️  No GPU available - test will be slow!")

    # Initialize CSM
    logger.info("\nInitializing CSM...")
    init_start = time.time()
    csm = CSMConverter()
    init_time = time.time() - init_start
    logger.info(f"Initialization took: {init_time:.2f}s")

    # Test text-to-audio
    test_text = "Hello! This is a test of the conversational speech model running on GPU."
    logger.info(f"\n🎤 Test: Converting text to audio")
    logger.info(f"Text: {test_text}")

    test_start = time.time()
    audio_path = csm.text_to_audio(test_text, output_path="test_csm_output.wav")
    test_time = time.time() - test_start

    logger.info(f"\n✓ Audio saved to: {audio_path}")
    logger.info(f"✓ Generation took: {test_time:.2f}s")
    
    # Performance assessment
    if test_time < 3.0:
        logger.info("✅ EXCELLENT - GPU is working properly!")
    elif test_time < 10.0:
        logger.warning("⚠️  SLOW - Check GPU utilization")
    else:
        logger.error("❌ VERY SLOW - CSM likely running on CPU!")
    
    # Final GPU stats
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        logger.info(f"\nGPU Memory Used: {allocated:.2f}GB")
    
    logger.info("\n" + "="*60)
    logger.info("✓ CSM integration test complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
