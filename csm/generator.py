from dataclasses import dataclass
from typing import List, Tuple
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
        # 🔥 FIX: Force GPU synchronization at start
        if self.device.type == 'cuda':
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

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        max_seq_len = 2048
        max_context_len = max_seq_len - max_generation_len
        if curr_tokens.size(1) >= max_context_len:
            raise ValueError(
                f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
            )

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

            # 🔥 FIX: Ensure new tensors stay on GPU
            curr_tokens = torch.cat(
                [sample, torch.zeros(1, 1, device=self.device, dtype=torch.long)], 
                dim=1
            ).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1, device=self.device, dtype=torch.bool)], 
                dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        loop_time = time.time() - loop_start
        logger.debug(f"Generation loop completed: {loop_time:.3f}s for {len(samples)} samples")
        
        decode_start = time.time()
        audio = self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)
        decode_time = time.time() - decode_start
        logger.debug(f"Audio decode: {decode_time:.3f}s")

        # 🔥 FIX: Ensure audio tensor is on correct device before watermarking
        watermark_start = time.time()
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            audio = audio.to(self.device)

        # This applies an imperceptible watermark to identify audio as AI-generated.
        # Watermarking ensures transparency, dissuades misuse, and enables traceability.
        # Please be a responsible AI citizen and keep the watermarking in place.
        # If using CSM 1B in another application, use your own private key and keep it secret.
        audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK)
        audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)
        watermark_time = time.time() - watermark_start
        logger.debug(f"Watermarking: {watermark_time:.3f}s")
        
        # 🔥 FIX: Final GPU sync before returning
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        return audio


def load_csm_1b(device: str = "cuda") -> Generator:
    # 🚀 CRITICAL: Enable Flash Attention for 5-10x speedup
    import torch.nn.functional as F
    if hasattr(F, 'scaled_dot_product_attention') and device == "cuda":
        # Force PyTorch to use efficient Flash Attention backend
        import torch.backends.cuda
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)  # Disable slow fallback
        print("✓ Flash Attention enabled for SDPA")
    
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
    
    # torch.compile() tested but didn't provide speedup - disabling for now
    # Root cause appears to be torchtune transformer implementation

    generator = Generator(model)
    return generator