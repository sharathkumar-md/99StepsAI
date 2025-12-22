"""
Conversational Chatbot with Dual-LLM Pipeline

Pipeline Flow:
1. User Input
2. LLM Response Generation (llama3.2) → Generates conversational response
3. CSM Text-to-Speech → Converts response to natural audio
4. Whisper Speech-to-Text → Transcribes audio (may add fillers)
5. LLM Text Cleanup (llama3.2) → Cleans transcription
6. Final Clean Output

This ensures conversational quality from CSM while maintaining clean text output.
"""

# Suppress warnings before imports
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import logging
import time
import numpy as np
from pathlib import Path

from csm_integration import CSMConverter
from asr_whisper import WhisperASR
from conversation_llm import ConversationLLM

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


class ConversationalChatbot:
    """
    Conversational chatbot with Dual-LLM + CSM pipeline

    Complete Flow:
    User input → LLM generates response → CSM audio → Whisper transcription → LLM cleanup → Clean output

    Features:
    - Conversational AI with memory
    - Natural audio quality from CSM
    - Clean text output
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
        logger.info("Loading CSM...")
        self.csm = CSMConverter(device=csm_device)

        logger.info("Loading Whisper...")
        self.whisper = WhisperASR(model_size=whisper_model_size)

        logger.info("Loading Dual-Mode LLM (llama3.2)...")
        logger.info("  - Mode 1: Response Generation")
        logger.info("  - Mode 2: Text Cleanup")
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
        Process user input through complete dual-LLM pipeline

        Pipeline: User → LLM Response → CSM Audio → Whisper → LLM Cleanup → Output

        Args:
            user_input: User's message (text) or path to audio file
            input_type: "text" or "audio"

        Returns:
            Chatbot's clean conversational response
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
        logger.info("\n[STEP 1/4] LLM Response Generation (llama3.2)")
        logger.info("-"*60)
        logger.info("Generating conversational response...")
        
        llm_gen_start = time.time()
        llm_response = self.llm.generate_response(user_text)
        llm_gen_time = time.time() - llm_gen_start

        logger.info("="*60)
        logger.info(f"💬 LLM RESPONSE: '{llm_response}' (took {llm_gen_time:.2f}s)")
        logger.info("="*60)

        # Step 3: CSM converts response to natural audio
        logger.info("\n[STEP 2/4] CSM Audio Generation")
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
        logger.info("\n[STEP 3/4] Whisper Transcription")
        logger.info("-"*60)
        logger.info("Transcribing CSM audio...")

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

        # Step 5: LLM cleans Whisper transcription
        logger.info("\n[STEP 4/4] LLM Text Cleanup (llama3.2)")
        logger.info("-"*60)
        logger.info("Cleaning Whisper transcription...")

        cleanup_start = time.time()
        final_response = self.llm.cleanup_text(whisper_output)
        cleanup_time = time.time() - cleanup_start
        
        # 🔥 Calculate total request time
        total_time = time.time() - request_start

        logger.info("="*60)
        logger.info("✓ FINAL RESPONSE READY")
        logger.info("="*60)
        logger.info(f"🎯 CLEAN OUTPUT: '{final_response}'")
        logger.info("")
        logger.info(f"⏱️  TIMING BREAKDOWN:")
        logger.info(f"   LLM Response:  {llm_gen_time:.2f}s")
        logger.info(f"   CSM Audio:     {csm_time:.2f}s")
        logger.info(f"   Whisper:       {whisper_time:.2f}s")
        logger.info(f"   LLM Cleanup:   {cleanup_time:.2f}s")
        logger.info(f"   TOTAL:         {total_time:.2f}s")
        logger.info("="*60)

        return final_response

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
        logger.info("Pipeline: User → LLM Response → CSM Audio → Whisper → LLM Cleanup → Clean Output")
        logger.info("Dual-Mode: Response Generation + Text Cleanup")
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
