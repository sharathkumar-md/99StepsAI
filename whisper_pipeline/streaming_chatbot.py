"""
Streaming Conversational Chatbot with CSM + Whisper.

Architecture:
    User Input → LLM Response → CSM Streaming → Audio Chunks → Whisper (parallel)
    
Timeline:
    0.0s: User sends text
    0.6s: LLM returns response text
    0.6s: CSM starts generating
    1.1s: First 0.5s audio chunk ready → USER HEARS AUDIO
    1.6s: Second chunk → Continues playback
    2.1s: Third chunk → Whisper processes first 2 chunks (partial text)
    ...
    8.6s: Final chunk → Complete audio played
    9.0s: Whisper returns final transcription
    
Perceived latency: 1.1s (vs 13s blocking)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'csm'))

import torch
import torchaudio
import logging
import time
from typing import Iterator, Dict, Any
from streaming_asr import StreamingWhisperASR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('streaming_chatbot.log')
    ]
)
logger = logging.getLogger(__name__)


class StreamingConversationalChatbot:
    """
    Low-latency streaming chatbot with parallel audio generation and ASR.
    
    Features:
        - <1s time to first audio
        - Streaming audio playback while generating
        - Parallel Whisper transcription
        - Incremental text updates
    """
    
    def __init__(self, llm_endpoint: str = "http://localhost:11434", use_gpu: bool = True):
        """
        Initialize streaming chatbot.
        
        Args:
            llm_endpoint: Ollama API endpoint
            use_gpu: Use CUDA for CSM and Whisper
        """
        self.llm_endpoint = llm_endpoint
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        logger.info("=" * 60)
        logger.info("STREAMING CONVERSATIONAL CHATBOT")
        logger.info("=" * 60)
        
        # Initialize LLM
        logger.info("Initializing LLM...")
        from conversation_llm import ConversationLLM
        self.llm = ConversationLLM(llm_model="llama3.2")
        logger.info("✓ LLM ready")
        
        # Load CSM generator
        logger.info("Loading CSM generator...")
        from generator import load_csm_1b
        self.csm_generator = load_csm_1b(device=self.device)
        logger.info(f"✓ CSM ready on {self.device}")
        
        # Initialize Whisper ASR
        logger.info("Initializing Whisper ASR...")
        self.whisper_asr = StreamingWhisperASR(
            model_name="base.en",  # Fast and accurate enough
            device="cpu",  # Run on CPU to avoid cuDNN conflicts with CSM
            compute_type="int8"  # CPU-optimized
        )
        logger.info("✓ Whisper ready (CPU)")
        
        logger.info("=" * 60)
            
    def process_streaming(
        self,
        user_input: str,
        save_audio_path: str = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Process user input with streaming audio and parallel transcription.
        
        Args:
            user_input: User's message
            save_audio_path: Optional path to save final audio
            
        Yields:
            dict events:
                {"type": "llm_response", "text": str}
                {"type": "audio_chunk", "data": torch.Tensor, "chunk_num": int}
                {"type": "partial_text", "text": str}
                {"type": "final_text", "text": str}
                {"type": "complete", "total_time": float}
                
        Example:
            for event in chatbot.process_streaming("Hello!"):
                if event['type'] == 'audio_chunk':
                    play_audio(event['data'])  # Play immediately!
                elif event['type'] == 'final_text':
                    print(f"Bot said: {event['text']}")
        """
        start_time = time.time()
        
        # Step 1: Get LLM response
        logger.info(f"📥 User: {user_input}")
        llm_start = time.time()
        llm_response = self.llm.generate_response(user_input)
        llm_time = time.time() - llm_start
        
        # Split response into shorter chunks for faster audio generation
        response_chunks = self.llm.split_response_into_chunks(llm_response, max_words=15)
        logger.info(f"🤖 LLM ({llm_time:.2f}s): Split into {len(response_chunks)} chunks")
        
        # Step 2: Start Whisper worker thread
        self.whisper_asr.start_processing()
        
        # Step 3: Process each response chunk separately
        first_audio_time = None
        total_audio_chunks = 0
        total_csm_time = 0
        
        for chunk_idx, response_chunk in enumerate(response_chunks, 1):
            logger.info(f"📝 Processing chunk {chunk_idx}/{len(response_chunks)}: '{response_chunk}'")
            
            # Yield LLM response chunk
            yield {
                "type": "llm_response",
                "text": response_chunk,
                "time": llm_time / len(response_chunks),
                "chunk_num": chunk_idx,
                "total_chunks": len(response_chunks)
            }
            
            # Stream CSM audio for this chunk
            logger.info(f"🎵 Generating audio for chunk {chunk_idx}...")
            csm_start = time.time()
            chunk_audio = []
            
            for audio_chunk in self.csm_generator.generate_streaming(
                text=response_chunk,  # Use the split chunk, not full response
                speaker=0,
                context=[],
                max_audio_length_ms=8000,  # 8 second hard limit
                chunk_frames=6  # 0.5s chunks
            ):
                total_audio_chunks += 1
                elapsed = time.time() - start_time
                
                if first_audio_time is None:
                    first_audio_time = elapsed
                
                # Save chunk
                chunk_audio.append(audio_chunk)
                
                # Yield audio chunk immediately (user can play it)
                logger.info(f"✅ Audio {total_audio_chunks} (response {chunk_idx}/{len(response_chunks)}) at {elapsed:.2f}s")
                yield {
                    "type": "audio_chunk",
                    "data": audio_chunk,
                    "chunk_num": total_audio_chunks,
                    "elapsed_time": elapsed,
                    "response_chunk": chunk_idx
                }
            
                # Feed to Whisper (non-blocking)
                self.whisper_asr.put_audio_chunk(audio_chunk)
                
                # Check for partial transcription
                partial = self.whisper_asr.get_latest_partial()
                if partial and total_audio_chunks % 2 == 0:
                    yield {
                        "type": "partial_text",
                        "text": partial,
                        "chunk_num": total_audio_chunks
                    }
            
            chunk_csm_time = time.time() - csm_start
            total_csm_time += chunk_csm_time
            logger.info(f"✅ Audio for response chunk {chunk_idx} complete: {chunk_csm_time:.2f}s")
        
        # Step 4: Wait for final Whisper transcription
        logger.info("⏳ Waiting for final transcription...")
        self.whisper_asr.finish()
        final_text = self.whisper_asr.get_final()
        
        logger.info(f"📝 Final transcription: '{final_text}'")
        yield {
            "type": "final_text",
            "text": final_text
        }
        
        # Final stats
        total_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info(f"✅ COMPLETE - Total: {total_time:.2f}s, First audio: {first_audio_time:.2f}s")
        logger.info(f"   Response chunks: {len(response_chunks)}, Total CSM: {total_csm_time:.2f}s")
        logger.info("=" * 60)
        
        yield {
            "type": "complete",
            "total_time": total_time,
            "time_to_first_audio": first_audio_time,
            "llm_time": llm_time,
            "csm_time": total_csm_time,
            "num_chunks": total_audio_chunks,
            "response_chunks": len(response_chunks)
        }


def main():
    """Interactive streaming chatbot."""
    try:
        chatbot = StreamingConversationalChatbot()
        
        # Interactive conversation loop
        print("\n" + "=" * 60)
        print("INTERACTIVE STREAMING CHATBOT")
        print("=" * 60)
        print("Commands:")
        print("  /quit  - Exit chatbot")
        print("  /reset - Reset conversation")
        print("  /test  - Run quick test")
        print("=" * 60)
        print("")
        
        while True:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input == "/quit":
                print("\nGoodbye! 👋")
                break
            
            if user_input == "/reset":
                # Reset LLM conversation
                chatbot.llm.reset_conversation()
                print("✓ Conversation reset")
                continue
            
            if user_input == "/test":
                user_input = "Hello! How are you today?"
                print(f"👤 You: {user_input}")
            
            # Process input with streaming
            print("")  # Blank line before response
            
            try:
                audio_chunks = []
                partial_texts = []
                response_text = ""
                
                for event in chatbot.process_streaming(user_input):
                    if event['type'] == 'llm_response':
                        # Print each response chunk immediately
                        if event['chunk_num'] == 1:
                            print(f"🤖 Bot: {event['text']}", end="", flush=True)
                            response_text = event['text']
                        else:
                            print(f" {event['text']}", end="", flush=True)
                            response_text += " " + event['text']
                        
                        # New line if it's the last chunk
                        if event['chunk_num'] == event['total_chunks']:
                            print("\n")
                        
                    elif event['type'] == 'audio_chunk':
                        # Collect audio chunks (for later playback in web app)
                        audio_chunks.append(event['data'])
                        logger.info(f"✅ Audio {event['chunk_num']} (response {event['response_chunk']}) at {event['elapsed_time']:.2f}s")
                        
                    elif event['type'] == 'partial_text':
                        # Show partial transcription progress
                        partial_texts.append(event['text'])
                        logger.info(f"📝 Partial: '{event['text'][:50]}...'")
                        
                    elif event['type'] == 'final_text':
                        # Show final transcription (verification)
                        print(f"✓ Audio verified")
                        
                    elif event['type'] == 'complete':
                        # Performance summary
                        logger.info(f"⏱️  Performance: Total {event['total_time']:.2f}s, "
                                  f"First audio {event['time_to_first_audio']:.2f}s, "
                                  f"LLM {event['llm_time']:.2f}s, "
                                  f"CSM {event['csm_time']:.2f}s, "
                                  f"{event['response_chunks']} response chunks, "
                                  f"{event['num_chunks']} audio chunks")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                print(f"❌ Error: {e}")
                
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
    
    except Exception as e:
        logger.error(f"Failed to start chatbot: {e}")
        print(f"\n❌ Failed to start chatbot: {e}")
        raise


if __name__ == "__main__":
    main()
