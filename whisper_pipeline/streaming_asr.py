"""
Streaming Whisper ASR with parallel audio processing.
Consumes audio chunks incrementally without blocking CSM generation.
"""

import torch
import queue
import threading
import logging
from typing import Iterator, Optional
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class StreamingWhisperASR:
    """
    Non-blocking Whisper ASR that processes audio chunks in parallel.
    
    Architecture:
        Main Thread: CSM generates audio → puts chunks in queue
        Worker Thread: Consumes queue → runs Whisper → yields transcriptions
        
    Benefits:
        - CSM doesn't wait for Whisper
        - User hears audio immediately
        - Transcription happens in parallel
    """
    
    def __init__(
        self,
        model_name: str = "base.en",
        device: str = "cuda",
        compute_type: str = "float16"
    ):
        """
        Initialize Whisper model for streaming ASR.
        
        Args:
            model_name: Whisper model size (tiny.en, base.en, small.en, medium.en)
            device: cuda or cpu
            compute_type: float16 (GPU) or int8 (CPU)
        """
        logger.info(f"Loading Whisper model: {model_name} on {device}")
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.device = device
        
        # Queue for audio chunks (main thread → worker thread)
        self.audio_queue = queue.Queue(maxsize=20)  # Bounded to prevent memory issues
        
        # Results
        self.partial_transcriptions = []
        self.final_transcription = ""
        
        # Threading
        self.worker_thread = None
        self.is_running = False
        
        # Audio accumulator
        self.accumulated_audio = []
        self.sample_rate = 24000
        
    def start_processing(self):
        """Start the worker thread for parallel ASR."""
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._process_audio_worker, daemon=True)
        self.worker_thread.start()
        logger.info("✓ Whisper worker thread started")
        
    def put_audio_chunk(self, audio_chunk: torch.Tensor):
        """
        Add audio chunk to processing queue (non-blocking).
        
        Args:
            audio_chunk: Audio tensor from CSM (24kHz, mono)
        """
        try:
            self.audio_queue.put(audio_chunk, timeout=1.0)
        except queue.Full:
            logger.warning("⚠️  Audio queue full, dropping chunk (backpressure)")
            
    def finish(self):
        """Signal end of audio and wait for final transcription."""
        self.audio_queue.put(None)  # Sentinel value
        if self.worker_thread:
            self.worker_thread.join(timeout=10.0)
        logger.info("✓ Whisper processing complete")
        
    def _process_audio_worker(self):
        """
        Worker thread: consume audio chunks and run Whisper.
        Yields partial transcriptions every 2 chunks (~1 second of audio).
        """
        chunk_count = 0
        
        while self.is_running:
            try:
                # Get audio chunk from queue (blocking)
                audio_chunk = self.audio_queue.get(timeout=0.5)
                
                # Check for end sentinel
                if audio_chunk is None:
                    break
                    
                # Accumulate audio
                self.accumulated_audio.append(audio_chunk.cpu())
                chunk_count += 1
                
                # Process every 2 chunks (~1s of audio) for partial transcription
                if chunk_count % 2 == 0:
                    self._transcribe_accumulated(partial=True)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Whisper worker error: {e}")
                
        # Final transcription with all accumulated audio
        if self.accumulated_audio:
            self._transcribe_accumulated(partial=False)
            
        self.is_running = False
        
    def _transcribe_accumulated(self, partial: bool = False):
        """
        Transcribe accumulated audio chunks.
        
        Args:
            partial: If True, this is a partial transcription (user feedback)
                    If False, this is the final transcription (complete audio)
        """
        try:
            # Concatenate all accumulated audio
            full_audio = torch.cat(self.accumulated_audio, dim=0)
            
            # Convert to numpy for Whisper (required format)
            audio_np = full_audio.numpy()
            
            # Resample to 16kHz (Whisper requirement)
            import torchaudio
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
            audio_16k = torchaudio.functional.resample(
                audio_tensor, 
                orig_freq=self.sample_rate, 
                new_freq=16000
            ).squeeze(0).numpy()
            
            # Run Whisper
            segments, info = self.model.transcribe(
                audio_16k,
                language="en",
                beam_size=1,  # Fast decoding
                vad_filter=True,  # Remove silence
                word_timestamps=False  # Don't need word-level timing
            )
            
            # Collect transcription
            transcription = " ".join([segment.text for segment in segments]).strip()
            
            if partial:
                self.partial_transcriptions.append(transcription)
                logger.debug(f"📝 Partial transcription: '{transcription[:50]}...'")
            else:
                self.final_transcription = transcription
                logger.info(f"✅ Final transcription: '{transcription}'")
                
        except Exception as e:
            logger.error(f"❌ Transcription error: {e}")
            
    def get_latest_partial(self) -> Optional[str]:
        """Get most recent partial transcription."""
        return self.partial_transcriptions[-1] if self.partial_transcriptions else None
        
    def get_final(self) -> str:
        """Get final complete transcription."""
        return self.final_transcription


def transcribe_streaming_audio(audio_chunks: Iterator[torch.Tensor]) -> dict:
    """
    Convenience function: transcribe streaming audio chunks in parallel.
    
    Args:
        audio_chunks: Iterator yielding audio tensors from CSM
        
    Returns:
        dict with 'partial_transcriptions' (list) and 'final_transcription' (str)
        
    Example:
        audio_chunks = csm_generator.generate_streaming("Hello", speaker=0, context=[])
        result = transcribe_streaming_audio(audio_chunks)
        print(result['final_transcription'])
    """
    asr = StreamingWhisperASR()
    asr.start_processing()
    
    # Feed chunks to ASR (non-blocking)
    for chunk in audio_chunks:
        asr.put_audio_chunk(chunk)
        
    # Wait for final transcription
    asr.finish()
    
    return {
        'partial_transcriptions': asr.partial_transcriptions,
        'final_transcription': asr.final_transcription
    }
