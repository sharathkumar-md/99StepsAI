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
                "You are a friendly, helpful voice assistant having a natural conversation.\n\n"
                "CONVERSATIONAL STYLE:\n"
                "- Talk naturally like a friend, not a formal assistant\n"
                "- Keep responses SHORT (1-3 sentences) since this is voice conversation\n"
                "- Use everyday language and contractions (I'm, you're, that's)\n"
                "- Be warm and personable - respond to emotions naturally\n"
                "- NEVER say 'as an AI' or 'as a language model' - just be helpful\n\n"
                "RESPONSE GUIDELINES:\n"
                "- Answer directly without unnecessary context or caveats\n"
                "- If asking a follow-up, make it conversational and brief\n"
                "- Match the user's energy and tone\n"
                "- For greetings, keep it simple and friendly\n"
                "- For questions, give the key answer first, then optional details\n\n"
                "EXAMPLES:\n"
                "User: 'Hello!'\n"
                "You: 'Hey! How can I help you today?'\n\n"
                "User: 'I'm feeling stressed'\n"
                "You: 'I hear you. Want to talk about what's on your mind?'\n\n"
                "User: 'What's the weather?'\n"
                "You: 'I can't check live weather, but you can try asking Siri or checking your weather app!'\n\n"
                "Keep it natural, brief, and conversational."
            )
        else:
            self.response_prompt = response_system_prompt

        # System prompt for TEXT CLEANUP (Mode 2)
        if cleanup_system_prompt is None:
            self.cleanup_prompt = (
                "You are a strict text normalizer for speech transcripts. Your sole job is to RETURN a cleaned version of the input text — do NOT add, respond, or rewrite meaning.\n\n"
                "PRINCIPLES:\n"
                "1) PRESERVE MEANING: Do not change the speaker's intent or add content.\n"
                "2) REMOVE DISFLUENCIES: Remove filler words (um, uh, like, you know), false starts, and stutters.\n"
                "3) CORRECT TEXT: Fix obvious ASR errors, punctuation, capitalization, and repeated words.\n\n"
                "STRICT RULES (follow exactly):\n"
                "- Output ONLY the cleaned text. No explanation, no commentary, no extra sentences.\n"
                "- Do NOT answer questions or turn the text into a response.\n"
                "- Do NOT change pronouns or the speaker perspective.\n"
                "- Do NOT summarize or shorten content unless removing disfluency.\n\n"
                "EXAMPLES:\n"
                "Input: 'um, I, uh, I don't know, I guess I'm tired'\n"
                "Output: 'I don't know, I guess I'm tired.'\n\n"
                "Input: 'wanna talk about what's not exciting'\n"
                "Output: 'Want to talk about what's not exciting?'\n\n"
                "Return only the cleaned text, in the same language as the input."
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

            # Check if model exists (llama3.2 might be listed as llama3.2:latest)
            if 'models' in models and isinstance(models['models'], list):
                available_models = [m.get('name', '').replace(':latest', '') for m in models['models']]
                model_found = any(self.llm_model in m or m in self.llm_model for m in available_models)
                if not model_found:
                    logger.warning(f"Model {self.llm_model} not found locally")
                    logger.warning(f"Available models: {', '.join(available_models)}")
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
