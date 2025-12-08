"""
Conversational Chatbot
Full pipeline: User input → CSM → Audio → Whisper → LLM Text Cleanup → Clean Text
CSM provides conversational quality, LLM cleans Whisper transcription
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
    Conversational chatbot with CSM pipeline
    Flow: User input → CSM → Audio → Whisper → LLM Cleanup → Clean Text
    CSM provides conversational audio, LLM cleans up Whisper transcription
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

        logger.info("Loading LLM (llama3.2) for text cleanup...")
        self.llm = ConversationLLM(
            llm_model=llm_model,
            system_prompt=system_prompt
        )

        logger.info("="*60)
        logger.info("✓ Chatbot Ready!")
        logger.info("="*60)
        logger.info("")

    def process_user_input(self, user_input: str, input_type: str = "text") -> str:
        """
        Process user input through pipeline and get response

        Args:
            user_input: User's message (text) or path to audio file
            input_type: "text" or "audio"

        Returns:
            Chatbot's response text
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
            logger.info("Transcribing audio with Whisper...")
            transcription = self.whisper.transcribe(user_input, verbose=False)
            user_text = transcription["text"].strip()
            logger.info(f"Transcribed: '{user_text}'")

        else:
            raise ValueError(f"Invalid input_type: {input_type}")

        # Step 2: Route through CSM
        logger.info("\n[STEP 1/3] CSM Audio Generation")
        logger.info("-"*60)
        logger.info(f"Converting text to audio via CSM...")

        csm_audio_path = self.csm.text_to_audio(
            text=user_text,
            output_path=None  # Use temp file
        )
        logger.info(f"✓ CSM audio: {csm_audio_path}")

        # Step 3: Transcribe CSM audio with Whisper
        logger.info("\n[STEP 2/3] Whisper Transcription")
        logger.info("-"*60)
        logger.info("Transcribing CSM audio with Whisper...")

        transcription = self.whisper.transcribe(
            csm_audio_path,
            verbose=False
        )
        processed_text = transcription["text"].strip()

        logger.info("="*60)
        logger.info(f"📝 WHISPER OUTPUT: '{processed_text}'")
        logger.info("="*60)

        # Cleanup temp CSM audio
        try:
            os.unlink(csm_audio_path)
        except:
            pass

        # Step 4: Clean and structure text with LLM
        logger.info("\n[STEP 3/3] LLM Text Cleanup (llama3.2)")
        logger.info("-"*60)
        logger.info("Cleaning and structuring Whisper output...")

        response = self.llm.chat(processed_text)

        logger.info("="*60)
        logger.info("✓ CLEANED TEXT GENERATED")
        logger.info("="*60)
        logger.info(f"🎯 FINAL OUTPUT: '{response}'")
        logger.info("="*60)

        return response

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
        logger.info("Pipeline: User → CSM (conversational audio) → Whisper → LLM (text cleanup) → Clean Text")
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
