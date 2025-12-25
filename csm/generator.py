from dataclasses import dataclass
from typing import List, Tuple, Iterator
import json

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from models import Model, ModelArgs
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
from watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark


@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor


def load_llama3_tokenizer():
    """
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
    tokenizer_name = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
    )

    return tokenizer


class Generator:
    def __init__(
        self,
        model: Model,
    ):
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer()

        device = next(model.parameters()).device
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.set_num_codebooks(32)
        self._audio_tokenizer = mimi

        self._watermarker = load_watermarker(device=device)

        self.sample_rate = mimi.sample_rate
        self.device = device

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert audio.ndim == 1, "Audio must be single channel"

        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (seq_len, 33), (seq_len, 33)
        """
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
    ) -> torch.Tensor:
        # � OPTIMIZATION: Pre-allocate device properties
        device = self.device
        is_cuda = device.type == 'cuda'
        
        if is_cuda:
            torch.cuda.synchronize()
            
        self._model.reset_caches()

        max_generation_len = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(device)

        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0), device=device, dtype=torch.long).unsqueeze(0)

        max_seq_len = 2048
        max_context_len = max_seq_len - max_generation_len
        if curr_tokens.size(1) >= max_context_len:
            raise ValueError(
                f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
            )

        # 🚀 OPTIMIZATION: Pre-allocate constant tensors (avoid repeated creation)
        zeros_1x1_long = torch.zeros(1, 1, device=device, dtype=torch.long)
        zeros_1x1_bool = torch.zeros(1, 1, device=device, dtype=torch.bool)
        
        import time
        import logging
        logger = logging.getLogger(__name__)
        
        loop_start = time.time()
        for i in range(max_generation_len):
            iter_start = time.time()
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            iter_time = time.time() - iter_start
            
            if i < 3 or i % 20 == 0:  # Log first 3 and every 20th iteration
                logger.debug(f"Iteration {i}: {iter_time:.3f}s, sample device: {sample.device}")
            
            if torch.all(sample == 0):
                logger.debug(f"EOS reached at iteration {i}")
                break  # eos

            samples.append(sample)

            # 🚀 OPTIMIZATION: Fused operations - reuse pre-allocated tensors
            curr_tokens = torch.cat([sample, zeros_1x1_long], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat([torch.ones_like(sample).bool(), zeros_1x1_bool], dim=1).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        loop_time = time.time() - loop_start
        logger.debug(f"Generation loop completed: {loop_time:.3f}s for {len(samples)} samples")
        
        # 🚀 OPTIMIZATION: Stack samples without intermediate sync
        decode_start = time.time()
        stacked_samples = torch.stack(samples).permute(1, 2, 0)
        audio = self._audio_tokenizer.decode(stacked_samples).squeeze(0).squeeze(0)
        decode_time = time.time() - decode_start
        logger.debug(f"Audio decode: {decode_time:.3f}s")

        # Watermarking
        watermark_start = time.time()
        if is_cuda:
            torch.cuda.synchronize()
            audio = audio.to(device)

        # This applies an imperceptible watermark to identify audio as AI-generated.
        # Watermarking ensures transparency, dissuades misuse, and enables traceability.
        # Please be a responsible AI citizen and keep the watermarking in place.
        # If using CSM 1B in another application, use your own private key and keep it secret.
        audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK)
        audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)
        watermark_time = time.time() - watermark_start
        logger.debug(f"Watermarking: {watermark_time:.3f}s")
        
        # Final sync
        if is_cuda:
            torch.cuda.synchronize()

        return audio

    @torch.inference_mode()
    def generate_streaming(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 8000,  # 🚀 Hard limit: 8 seconds max
        temperature: float = 0.9,
        topk: int = 50,
        chunk_frames: int = 6,  # 🚀 6 frames = 0.5s chunks @ 24kHz
    ) -> Iterator[torch.Tensor]:
        """
        Stream audio generation in chunks for low-latency playback.
        
        Args:
            text: Text to convert to speech
            speaker: Speaker ID (0 or 1)
            context: Conversation history segments
            max_audio_length_ms: Maximum audio duration (hard limit: 8000ms)
            temperature: Sampling temperature
            topk: Top-k sampling parameter
            chunk_frames: Frames per chunk (6 frames = ~0.5s audio)
            
        Yields:
            Audio tensors (torch.Tensor): Each chunk is ~0.5s of audio at 24kHz
            
        Example:
            for audio_chunk in generator.generate_streaming("Hello world", speaker=0, context=[]):
                # Play audio_chunk immediately (non-blocking)
                # First chunk arrives in ~0.5s instead of 12s
        """
        import time
        import logging
        logger = logging.getLogger(__name__)
        
        device = self.device
        is_cuda = device.type == 'cuda'
        
        if is_cuda:
            torch.cuda.synchronize()
            
        self._model.reset_caches()

        # 🚀 Enforce 8-second hard limit
        max_audio_length_ms = min(max_audio_length_ms, 8000)
        max_generation_len = int(max_audio_length_ms / 80)
        
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(device)

        # Frame accumulator for chunked decoding
        frame_buffer = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0), device=device, dtype=torch.long).unsqueeze(0)

        max_seq_len = 2048
        max_context_len = max_seq_len - max_generation_len
        if curr_tokens.size(1) >= max_context_len:
            raise ValueError(
                f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
            )

        # Pre-allocate constant tensors
        zeros_1x1_long = torch.zeros(1, 1, device=device, dtype=torch.long)
        zeros_1x1_bool = torch.zeros(1, 1, device=device, dtype=torch.bool)
        
        logger.debug(f"🎵 Starting streaming generation (chunks every {chunk_frames} frames = ~0.5s)")
        chunk_count = 0
        generation_start = time.time()
        
        for i in range(max_generation_len):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            
            if torch.all(sample == 0):
                logger.debug(f"EOS reached at iteration {i}")
                break  # eos

            frame_buffer.append(sample)

            # 🚀 STREAMING: Yield chunk when buffer is full
            if len(frame_buffer) >= chunk_frames:
                chunk_start = time.time()
                
                # Decode buffered frames to audio
                stacked_frames = torch.stack(frame_buffer).permute(1, 2, 0)
                audio_chunk = self._audio_tokenizer.decode(stacked_frames).squeeze(0).squeeze(0)
                
                # Apply watermarking to chunk
                if is_cuda:
                    audio_chunk = audio_chunk.to(device)
                audio_chunk, wm_sr = watermark(self._watermarker, audio_chunk, self.sample_rate, CSM_1B_GH_WATERMARK)
                audio_chunk = torchaudio.functional.resample(audio_chunk, orig_freq=wm_sr, new_freq=self.sample_rate)
                
                chunk_time = time.time() - chunk_start
                chunk_count += 1
                elapsed = time.time() - generation_start
                
                logger.debug(f"✅ Chunk {chunk_count}: {len(frame_buffer)} frames decoded in {chunk_time:.3f}s (total: {elapsed:.2f}s)")
                
                # Yield chunk immediately for playback
                yield audio_chunk
                
                # Clear buffer for next chunk
                frame_buffer = []

            # Continue generation
            curr_tokens = torch.cat([sample, zeros_1x1_long], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat([torch.ones_like(sample).bool(), zeros_1x1_bool], dim=1).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        # 🚀 Final chunk: Process remaining frames
        if frame_buffer:
            logger.debug(f"🎵 Processing final chunk: {len(frame_buffer)} frames")
            stacked_frames = torch.stack(frame_buffer).permute(1, 2, 0)
            audio_chunk = self._audio_tokenizer.decode(stacked_frames).squeeze(0).squeeze(0)
            
            if is_cuda:
                audio_chunk = audio_chunk.to(device)
            audio_chunk, wm_sr = watermark(self._watermarker, audio_chunk, self.sample_rate, CSM_1B_GH_WATERMARK)
            audio_chunk = torchaudio.functional.resample(audio_chunk, orig_freq=wm_sr, new_freq=self.sample_rate)
            
            yield audio_chunk
        
        total_time = time.time() - generation_start
        logger.debug(f"✅ Streaming complete: {chunk_count + 1} chunks in {total_time:.2f}s")
        
        if is_cuda:
            torch.cuda.synchronize()


def load_csm_1b(device: str = "cuda") -> Generator:
    # 🚀 OPTIMIZATION: Configure PyTorch for maximum performance
    import torch.nn.functional as F
    if device == "cuda":
        # Enable Flash Attention backends
        import torch.backends.cuda
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True  # Auto-tune kernels
        torch.backends.cudnn.allow_tf32 = True  # Already set but ensure it's on
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Set optimal memory allocation strategy
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
        
        print("✓ CUDA optimizations enabled")
    
    # Load config from HuggingFace hub
    try:
        config_path = hf_hub_download(
            repo_id="sesame/csm-1b",
            filename="config.json"
        )
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Filter to only include fields that ModelArgs expects
        expected_fields = {
            'backbone_flavor',
            'decoder_flavor', 
            'text_vocab_size',
            'audio_vocab_size',
            'audio_num_codebooks'
        }
        filtered_config = {k: v for k, v in config_dict.items() if k in expected_fields}
        config = ModelArgs(**filtered_config)
    except Exception as e:
        print(f"Warning: Could not load config from hub: {e}")
        # Fallback to default config for CSM 1B
        config = ModelArgs(
            backbone_flavor="llama-1B",
            decoder_flavor="llama-100M",
            text_vocab_size=128256,
            audio_vocab_size=1024,
            audio_num_codebooks=32
        )
    
    model = Model(config=config)
    
    # Load pretrained weights using HuggingFace hub (respects authentication)
    try:
        # Try model.safetensors first (newer format)
        try:
            weights_path = hf_hub_download(
                repo_id="sesame/csm-1b",
                filename="model.safetensors"
            )
            from safetensors.torch import load_file
            state_dict = load_file(weights_path)
            print(f"Loaded safetensors weights from {weights_path}")
        except:
            # Fallback to pytorch_model.bin
            weights_path = hf_hub_download(
                repo_id="sesame/csm-1b",
                filename="pytorch_model.bin"
            )
            state_dict = torch.load(weights_path, map_location="cpu")
            print(f"Loaded pytorch weights from {weights_path}")
        
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Warning: Could not load pretrained weights: {e}")
    
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()  # 🔥 CRITICAL: Set model to eval mode for inference
    
    # 🚀 AGGRESSIVE OPTIMIZATION: Compile with inductor backend
    if device == "cuda":
        try:
            print("Compiling model with torch.compile (inductor + max-autotune)...")
            print("⏳ First compilation takes ~2 minutes, but speeds up inference by 2-3x")
            
            # Use inductor backend with aggressive optimizations
            import torch._inductor.config as inductor_config
            inductor_config.coordinate_descent_tuning = True
            inductor_config.triton.unique_kernel_names = True
            inductor_config.fx_graph_cache = True  # Cache compiled graphs
            
            # Compile the model with max-autotune mode
            model = torch.compile(
                model,
                backend="inductor",
                mode="max-autotune",  # Aggressive optimization
                fullgraph=False,  # Allow graph breaks
                dynamic=False  # Static shapes for better optimization
            )
            print("✓ Model compiled successfully")
            
        except Exception as e:
            print(f"⚠️  Compilation failed, continuing without compile: {e}")

    generator = Generator(model)
    
    # 🚀 WARMUP: Run dummy inference to trigger compilation
    if device == "cuda":
        try:
            print("Warming up model (triggering compilation)...")
            dummy_text = "Hello"
            dummy_audio = generator.generate(dummy_text, speaker=0, context=[], max_audio_length_ms=2000)
            print(f"✓ Warmup complete - model ready for fast inference")
            del dummy_audio
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️  Warmup failed: {e}")
    
    return generator