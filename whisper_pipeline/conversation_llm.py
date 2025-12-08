"""
Conversational LLM Module
Handles multi-turn conversations with LLM (llama3.2)
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
    """Conversational LLM with history management"""

    def __init__(
        self,
        llm_model: str = "llama3.2",
        system_prompt: str = None
    ):
        """
        Initialize conversational LLM

        Args:
            llm_model: LLM model to use (always llama3.2)
            system_prompt: System prompt for the conversation
        """
        load_dotenv()

        self.llm_model = llm_model
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

        # Default system prompt - text cleanup and structuring
        if system_prompt is None:
            self.system_prompt = (
                "You are a text cleanup assistant. Your ONLY job is to clean and structure transcribed text. "
                "The text comes from speech (via CSM → Whisper) and may contain filler sounds like 'umm', 'ohh', 'uh', etc. "
                "\n"
                "Your task:\n"
                "1. Remove ONLY filler sounds and stutters (umm, uh, ohh, ah, like, you know, etc.)\n"
                "2. Fix obvious transcription errors (spelling, punctuation)\n"
                "3. Keep the EXACT same meaning - do NOT rephrase, interpret, or add anything\n"
                "4. Preserve the original casual, conversational tone\n"
                "5. Return ONLY the cleaned text - nothing more, nothing less\n"
                "\n"
                "CRITICAL RULES:\n"
                "- You are NOT having a conversation\n"
                "- You are NOT generating new responses\n"
                "- You are ONLY removing filler words while keeping the EXACT meaning\n"
                "- If unsure, keep the original text unchanged\n"
                "- Preserve the speaker's intent, tone, emotion, and message EXACTLY as spoken"
            )
        else:
            self.system_prompt = system_prompt

        # Conversation history
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

        logger.info("Conversational LLM initialized")

    def chat(self, user_message: str) -> str:
        """
        Send message and get response

        Args:
            user_message: User's message

        Returns:
            Assistant's response
        """
        logger.info(f"User: {user_message}")

        # Add user message to history
        self.history.append({
            "role": "user",
            "content": user_message
        })

        # Prepare messages for API
        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.history

        try:
            # Get response from LLM
            logger.debug("Calling LLM...")
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

            logger.info(f"Assistant: {assistant_message}")

            return assistant_message

        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
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

    def set_system_prompt(self, prompt: str):
        """Update system prompt"""
        self.system_prompt = prompt
        logger.info("System prompt updated")


def main():
    """Test conversational LLM"""
    logger.info("="*60)
    logger.info("CONVERSATIONAL LLM TEST")
    logger.info("="*60)

    # Initialize LLM
    llm = ConversationLLM(llm_model="llama3.2")

    # Test conversation
    test_messages = [
        "Hello! How are you?",
        "What's your name?",
        "Can you tell me a joke?",
        "Thanks! Goodbye!"
    ]

    logger.info("\nStarting test conversation...\n")

    for message in test_messages:
        logger.info("-"*60)
        response = llm.chat(message)
        logger.info("")

    logger.info("-"*60)
    logger.info(f"\nConversation turns: {len(llm.history) // 2}")
    logger.info("✓ Conversational LLM test complete!")


if __name__ == "__main__":
    main()
