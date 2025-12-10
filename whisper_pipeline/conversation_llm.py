"""
Conversational LLM Module
Dual-mode LLM: Response Generation + Text Cleanup
Uses llama3.2 via Ollama
"""

import os
import logging
from typing import List, Dict
from dotenv import load_dotenv
import ollama

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


class ConversationLLM:
    """
    Dual-mode LLM for conversational AI

    Mode 1: Response Generation - Normal chatbot with conversation history
    Mode 2: Text Cleanup - Cleans Whisper transcriptions (no history)
    """

    def __init__(
        self,
        llm_model: str = "llama3.2",
        response_system_prompt: str = None,
        cleanup_system_prompt: str = None
    ):
        """
        Initialize dual-mode LLM

        Args:
            llm_model: LLM model to use (llama3.2)
            response_system_prompt: System prompt for response generation
            cleanup_system_prompt: System prompt for text cleanup
        """
        load_dotenv()

        self.llm_model = llm_model
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

        # System prompt for RESPONSE GENERATION (Mode 1)
        if response_system_prompt is None:
            self.response_prompt = (
                "You are a friendly, helpful assistant having a natural conversation.\n\n"
                "CONVERSATION GUIDELINES:\n"
                "- Be warm, personable, and genuine\n"
                "- Keep responses concise (2-3 sentences max)\n"
                "- Use contractions (I'm, you're, can't) for natural flow\n"
                "- Be conversational but clear\n"
                "- Match the user's energy and tone\n"
                "- Answer questions directly and helpfully\n"
                "- Remember context from earlier in the conversation\n\n"
                "IMPORTANT:\n"
                "- DON'T say 'I'm an AI' or 'I'm a language model'\n"
                "- Just be helpful without explaining what you are\n"
                "- Focus on answering the user's question naturally\n\n"
                "Be natural, be helpful, be conversational!"
            )
        else:
            self.response_prompt = response_system_prompt

        # System prompt for TEXT CLEANUP (Mode 2)
        if cleanup_system_prompt is None:
            self.cleanup_prompt = (
                "You normalize spoken text into clean written text.\n\n"
                "Remove filler sounds (um, uh, ah, you know).\n"
                "Fix obvious speech-to-text mistakes.\n"
                "Preserve the original meaning and intent exactly.\n"
                "Preserve the original conversational tone and natural flow.\n\n"
                "Do not add, remove, or rephrase content.\n"
                "Return only the cleaned text."
            )
        else:
            self.cleanup_prompt = cleanup_system_prompt

        # Conversation history (only used in response generation mode)
        self.history: List[Dict[str, str]] = []

        # Test connection
        try:
            ollama_client = ollama.Client(host=self.ollama_host)
            models = ollama_client.list()
            logger.info(f"Connected to Ollama at {self.ollama_host}")
            logger.info(f"Using LLM model: {self.llm_model}")

            # Check if model exists
            if 'models' in models and isinstance(models['models'], list):
                available_models = [m.get('name', '') for m in models['models']]
                if not any(self.llm_model in m for m in available_models):
                    logger.warning(f"Model {self.llm_model} not found locally")
            else:
                logger.warning("Could not retrieve model list")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama: {e}")
            logger.info("Make sure Ollama is running: ollama serve")

        logger.info("Dual-mode LLM initialized (Response Generation + Text Cleanup)")

    def generate_response(self, user_message: str) -> str:
        """
        MODE 1: Generate conversational response
        Uses conversation history for context

        Args:
            user_message: User's input message

        Returns:
            AI-generated conversational response
        """
        logger.info(f"[RESPONSE MODE] User: {user_message}")

        # Add user message to history
        self.history.append({
            "role": "user",
            "content": user_message
        })

        # Prepare messages with response generation prompt
        messages = [
            {"role": "system", "content": self.response_prompt}
        ] + self.history

        try:
            # Get response from LLM
            logger.debug("Calling LLM for response generation...")
            response = ollama.chat(
                model=self.llm_model,
                messages=messages
            )

            assistant_message = response['message']['content']

            # Add assistant response to history
            self.history.append({
                "role": "assistant",
                "content": assistant_message
            })

            logger.info(f"[RESPONSE MODE] Assistant: {assistant_message}")

            return assistant_message

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            logger.error("Make sure Ollama is running: ollama serve")
            logger.error(f"And model is available: ollama pull {self.llm_model}")
            raise

    def cleanup_text(self, whisper_text: str) -> str:
        """
        MODE 2: Clean Whisper transcription
        No conversation history - single-shot cleanup

        Args:
            whisper_text: Raw Whisper transcription (may contain fillers)

        Returns:
            Cleaned text (fillers removed, proper formatting)
        """
        logger.info(f"[CLEANUP MODE] Input: {whisper_text}")

        # Single message with cleanup prompt (no history)
        messages = [
            {"role": "system", "content": self.cleanup_prompt},
            {"role": "user", "content": whisper_text}
        ]

        try:
            # Get cleanup result from LLM
            logger.debug("Calling LLM for text cleanup...")
            response = ollama.chat(
                model=self.llm_model,
                messages=messages
            )

            cleaned_text = response['message']['content']

            logger.info(f"[CLEANUP MODE] Output: {cleaned_text}")

            return cleaned_text

        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            logger.error("Make sure Ollama is running: ollama serve")
            logger.error(f"And model is available: ollama pull {self.llm_model}")
            raise

    def reset_conversation(self):
        """Clear conversation history"""
        self.history = []
        logger.info("Conversation history cleared")

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.history.copy()


def main():
    """Test dual-mode LLM"""
    logger.info("="*60)
    logger.info("DUAL-MODE LLM TEST")
    logger.info("="*60)

    # Initialize LLM
    llm = ConversationLLM(llm_model="llama3.2")

    logger.info("\n--- Testing RESPONSE MODE ---\n")

    # Test response generation
    response1 = llm.generate_response("Hello! What's your name?")
    response2 = llm.generate_response("Tell me a joke!")

    logger.info("\n--- Testing CLEANUP MODE ---\n")

    # Test text cleanup
    dirty_text = "Um, hello there, uh, how are you, like, doing today?"
    clean_text = llm.cleanup_text(dirty_text)

    logger.info("\n" + "="*60)
    logger.info("✓ Dual-mode LLM test complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
