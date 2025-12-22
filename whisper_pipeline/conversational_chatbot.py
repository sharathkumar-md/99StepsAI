"""
Conversational Chatbot with LLM + CSM Pipeline

Pipeline Flow:
1. User Input
2. LLM Response Generation (llama3.2) → Generates conversational response
3. CSM Text-to-Speech → Converts response to natural audio
4. Whisper Speech-to-Text → Transcribes CSM audio output
5. Return Whisper Output → Direct CSM audio transcription

This preserves CSM's natural conversational audio characteristics in text form.
"""

import logging
import sys
import os

os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('logs/chatbot.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ],
    force=True   # REQUIRED
)

logger = logging.getLogger(__name__)
logger.info(" LOGGING INITIALIZED ")

import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
from pathlib import Path

os.environ["TQDM_DISABLE"] = "1"

from csm_integration import CSMConverter
from asr_whisper import WhisperASR
from conversation_llm import ConversationLLM

class ConversationalChatbot:
    """
    Conversational chatbot with LLM + CSM pipeline

    Pipeline:
    User input → LLM generates response → CSM audio → Whisper transcription → CSM output

    Features:
    - Conversational AI with memory
    - Natural audio quality from CSM
    - CSM audio characteristics preserved in text
    """

    def __init__(
        self,
        whisper_model_size: str = "base",
        llm_model: str = "llama3.2",
        csm_device: str = None,
        system_prompt: str = None
    ):
        """
        Initialize conversational chatbot

        Args:
            whisper_model_size: Whisper model size
            llm_model: LLM model (always llama3.2)
            csm_device: Device for CSM
            system_prompt: System prompt for conversation
        """
        logger.info("="*60)
        logger.info("INITIALIZING CONVERSATIONAL CHATBOT")
        logger.info("="*60)

        # Initialize components
        logger.info("Loading CSM (Conversational Speech Model)...")
        self.csm = CSMConverter(device=csm_device)

        logger.info("Loading Whisper (for CSM audio transcription)...")
        self.whisper = WhisperASR(model_size=whisper_model_size)

        logger.info("Loading LLM (llama3.2 for response generation)...")
        self.llm = ConversationLLM(
            llm_model=llm_model
        )

        # 🔥 Pre-warm all models (CRITICAL for fast first response)
        logger.info("\n" + "="*60)
        logger.info("PRE-WARMING MODELS (One-time setup)")
        logger.info("="*60)
        self._prewarm_models()

        logger.info("="*60)
        logger.info("✓ Chatbot Ready - All Models Loaded!")
        logger.info("="*60)
        logger.info("")

    def _prewarm_models(self):
        """
        Pre-warm all models by running dummy inference
        This eliminates 5-10 second delay on first real request
        
        Flow:
        1. CSM: Generate 1 second of dummy audio
        2. Whisper: Transcribe 1 second of silence
        3. LLM: Generate one dummy response and cleanup
        """
        total_start = time.time()
        
        try:
            # Detect GPU
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info(f"✓ GPU Detected: {torch.cuda.get_device_name(0)}")
                    logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                else:
                    logger.warning("⚠️  No GPU detected - running on CPU (slower)")
            except Exception as e:
                logger.debug(f"Could not check GPU: {e}")
            
            # 1. Pre-warm CSM (usually the slowest)
            logger.info("\n[1/3] Pre-warming CSM model...")
            csm_start = time.time()
            try:
                # Generate very short audio to load model into GPU
                dummy_csm_path = self.csm.text_to_audio(
                    text="Hi",
                    output_path=None
                )
                csm_time = time.time() - csm_start
                logger.info(f"✓ CSM ready in {csm_time:.2f}s")
                
                # Cleanup dummy file
                try:
                    os.unlink(dummy_csm_path)
                except:
                    pass
            except Exception as e:
                logger.warning(f"CSM pre-warming failed: {e}")
            
            # 2. Pre-warm Whisper
            logger.info("\n[2/3] Pre-warming Whisper model...")
            whisper_start = time.time()
            try:
                # Create 1 second of silence for dummy transcription
                dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 sec at 16kHz
                
                # Save to temp file
                import tempfile
                import soundfile as sf
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    dummy_audio_path = tmp.name
                    sf.write(dummy_audio_path, dummy_audio, 16000)
                
                # Transcribe to load model
                self.whisper.transcribe(dummy_audio_path, verbose=False)
                whisper_time = time.time() - whisper_start
                logger.info(f"✓ Whisper ready in {whisper_time:.2f}s")
                
                # Cleanup
                try:
                    os.unlink(dummy_audio_path)
                except:
                    pass
            except Exception as e:
                logger.warning(f"Whisper pre-warming failed: {e}")
            
            # 3. Pre-warm LLM (both modes)
            logger.info("\n[3/3] Pre-warming LLM models...")
            llm_start = time.time()
            try:
                # Test response generation
                self.llm.generate_response("Hi")
                # Test text cleanup
                self.llm.cleanup_text("um, hello there")
                # Clear the dummy conversation
                self.llm.reset_conversation()
                
                llm_time = time.time() - llm_start
                logger.info(f"✓ LLM ready in {llm_time:.2f}s")
            except Exception as e:
                logger.warning(f"LLM pre-warming failed: {e}")
            
            total_time = time.time() - total_start
            
            logger.info("\n" + "-"*60)
            logger.info(f"✓ Pre-warming complete in {total_time:.2f}s")
            logger.info("Next requests will be MUCH faster (1-3s instead of 10-15s)")
            logger.info("-"*60)
            
        except Exception as e:
            logger.warning(f"Pre-warming encountered errors: {e}")
            logger.warning("Models will load on first request instead")

    def process_user_input(self, user_input: str, input_type: str = "text") -> str:
        """
        Process user input through LLM + CSM pipeline

        Pipeline: User → LLM Response → CSM Audio → Whisper → CSM Output

        Args:
            user_input: User's message (text) or path to audio file
            input_type: "text" or "audio"

        Returns:
            CSM audio transcription (Whisper output)
        """
        # 🔥 Start total timing
        request_start = time.time()
        
        logger.info("\n" + "="*60)
        logger.info("PROCESSING USER INPUT")
        logger.info("="*60)

        # Step 1: Get user text
        if input_type == "text":
            logger.info(f"[INPUT] Text: '{user_input}'")
            user_text = user_input

        elif input_type == "audio":
            logger.info(f"[INPUT] Audio file: {user_input}")
            logger.info("Transcribing user audio with Whisper...")
            transcription = self.whisper.transcribe(user_input, verbose=False)
            user_text = transcription["text"].strip()
            logger.info(f"User transcription: '{user_text}'")

        else:
            raise ValueError(f"Invalid input_type: {input_type}")

        # Step 2: LLM generates conversational response
        logger.info("\n[STEP 1/3] LLM Response Generation (llama3.2)")
        logger.info("-"*60)
        logger.info("Generating conversational response...")
        
        llm_gen_start = time.time()
        llm_response = self.llm.generate_response(user_text)
        llm_gen_time = time.time() - llm_gen_start

        logger.info("="*60)
        logger.info(f"💬 LLM RESPONSE: '{llm_response}' (took {llm_gen_time:.2f}s)")
        logger.info("="*60)

        # Step 3: CSM converts response to natural audio
        logger.info("\n[STEP 2/3] CSM Audio Generation")
        logger.info("-"*60)
        logger.info("Converting LLM response to conversational audio...")

        csm_start = time.time()
        csm_audio_path = self.csm.text_to_audio(
            text=llm_response,
            output_path=None  # Use temp file
        )
        csm_time = time.time() - csm_start
        logger.info(f"✓ CSM audio: {csm_audio_path} (took {csm_time:.2f}s)")

        # Step 4: Whisper transcribes CSM audio
        logger.info("\n[STEP 3/3] Whisper Transcription of CSM Audio")
        logger.info("-"*60)
        logger.info("Transcribing CSM audio to get final response...")

        whisper_start = time.time()
        transcription = self.whisper.transcribe(
            csm_audio_path,
            verbose=False
        )
        whisper_output = transcription["text"].strip()
        whisper_time = time.time() - whisper_start

        logger.info("="*60)
        logger.info(f"📝 WHISPER OUTPUT: '{whisper_output}' (took {whisper_time:.2f}s)")
        logger.info("="*60)

        # If Whisper returns empty, CSM audio might be silent - use original LLM response
        if not whisper_output:
            logger.warning("⚠️  Whisper returned empty transcription!")
            logger.warning("CSM audio may be silent or corrupted.")
            logger.warning("Falling back to original LLM response.")
            whisper_output = llm_response

        # Cleanup temp CSM audio
        try:
            os.unlink(csm_audio_path)
        except:
            pass
        
        # 🔥 Calculate total request time
        total_time = time.time() - request_start

        logger.info("="*60)
        logger.info("✓ CSM RESPONSE READY")
        logger.info("="*60)
        logger.info(f"🎯 CSM OUTPUT (via Whisper): '{whisper_output}'")
        logger.info("")
        logger.info(f"⏱️  TIMING BREAKDOWN:")
        logger.info(f"   LLM Response:  {llm_gen_time:.2f}s")
        logger.info(f"   CSM Audio:     {csm_time:.2f}s")
        logger.info(f"   Whisper:       {whisper_time:.2f}s")
        logger.info(f"   TOTAL:         {total_time:.2f}s")
        logger.info("="*60)

        return whisper_output

    def chat(self, message: str) -> str:
        """
        Simple chat interface (text input)

        Args:
            message: User's message

        Returns:
            Chatbot's response
        """
        return self.process_user_input(message, input_type="text")

    def reset(self):
        """Reset conversation history"""
        self.llm.reset_conversation()
        logger.info("Conversation reset")


def main():
    """Interactive chatbot"""
    # Additional warning suppression for runtime
    import logging as builtin_logging
    builtin_logging.getLogger('torch._dynamo').setLevel(builtin_logging.ERROR)
    builtin_logging.getLogger('triton').setLevel(builtin_logging.ERROR)

    logger.info("\n" + "🤖 CONVERSATIONAL CHATBOT" + "\n")

    # Check venv
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )

    if not in_venv:
        logger.warning("⚠️  NOT RUNNING IN VENV!")
        logger.warning("Please activate venv first")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Initialize chatbot
    try:
        logger.info("Initializing chatbot...")
        logger.info("Pipeline: User → LLM Response → CSM Audio → Whisper → CSM Output")
        logger.info("Gets natural CSM audio characteristics in text form")
        logger.info("")

        chatbot = ConversationalChatbot(
            whisper_model_size="base",
            llm_model="llama3.2"
        )

        # Interactive conversation loop
        logger.info("\n" + "="*60)
        logger.info("CHAT MODE")
        logger.info("="*60)
        logger.info("Commands:")
        logger.info("  /quit  - Exit chatbot")
        logger.info("  /reset - Reset conversation")
        logger.info("="*60)
        logger.info("")

        while True:
            # Get user input
            user_input = input("\n👤 You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input == "/quit":
                logger.info("Goodbye!")
                break

            if user_input == "/reset":
                chatbot.reset()
                logger.info("Conversation reset")
                continue

            # Get response
            try:
                response = chatbot.chat(user_input)
                logger.info(f"\n🤖 Bot: {response}")
                

            except KeyboardInterrupt:
                logger.info("\n\nGoodbye!")
                break

            except Exception as e:
                logger.error(f"Error: {e}")

    except KeyboardInterrupt:
        logger.info("\n\nGoodbye!")

    except Exception as e:
        logger.error(f"\nFailed to start chatbot: {e}")
        logger.error("\nTroubleshooting:")
        logger.error("1. Activate venv: venv\\Scripts\\activate")
        logger.error("2. Start Ollama: ollama serve")
        logger.error("3. Verify llama3.2: ollama list")
        logger.error("4. Login to HuggingFace: huggingface-cli login")
        raise


if __name__ == "__main__":
    main()
