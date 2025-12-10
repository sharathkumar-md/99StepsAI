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

        logger.info("="*60)
        logger.info("✓ Chatbot Ready!")
        logger.info("="*60)
        logger.info("")

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

        llm_response = self.llm.generate_response(user_text)

        logger.info("="*60)
        logger.info(f"💬 LLM RESPONSE: '{llm_response}'")
        logger.info("="*60)

        # Step 3: CSM converts response to natural audio
        logger.info("\n[STEP 2/4] CSM Audio Generation")
        logger.info("-"*60)
        logger.info("Converting LLM response to conversational audio...")

        csm_audio_path = self.csm.text_to_audio(
            text=llm_response,
            output_path=None  # Use temp file
        )
        logger.info(f"✓ CSM audio: {csm_audio_path}")

        # Step 4: Whisper transcribes CSM audio
        logger.info("\n[STEP 3/4] Whisper Transcription")
        logger.info("-"*60)
        logger.info("Transcribing CSM audio...")

        transcription = self.whisper.transcribe(
            csm_audio_path,
            verbose=False
        )
        whisper_output = transcription["text"].strip()

        logger.info("="*60)
        logger.info(f"📝 WHISPER OUTPUT: '{whisper_output}'")
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

        final_response = self.llm.cleanup_text(whisper_output)

        logger.info("="*60)
        logger.info("✓ FINAL RESPONSE READY")
        logger.info("="*60)
        logger.info(f"🎯 CLEAN OUTPUT: '{final_response}'")
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
