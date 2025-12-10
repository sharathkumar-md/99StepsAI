# 99StepsAI - Voice AI Projects

Collection of AI projects focused on conversational AI and voice processing.

## Projects

### Whisper Pipeline - Conversational Chatbot 🤖

Full-featured AI chatbot with dual-LLM voice processing pipeline.

**Pipeline:** `User → LLM Response → CSM Audio → Whisper → LLM Cleanup → Clean Output`

**Features:**
- ✅ Dual-mode LLM (response generation + text cleanup)
- ✅ Natural conversation with context and history
- ✅ CSM audio quality for conversational tone
- ✅ Clean text output with fillers removed
- ✅ GPU acceleration support

[View Full Documentation →](whisper_pipeline/README.md)

**Quick Start:**
```bash
# Activate venv
venv\Scripts\activate

# Run chatbot
cd whisper_pipeline
python conversational_chatbot.py
```

**Architecture:**
```
1. User Input
   ↓
2. LLM (llama3.2) - Generates conversational response
   ↓
3. CSM - Converts to natural audio
   ↓
4. Whisper - Transcribes audio
   ↓
5. LLM (llama3.2) - Cleans transcription
   ↓
6. Clean Output ✨
```

### CSM (Conversational Speech Model)

Text-to-speech generation using CSM from Sesame AI Labs.

**Purpose:** Provides natural conversational audio quality in the pipeline.

[View CSM Documentation →](csm/README.md)

## Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/99StepsAI.git
cd 99StepsAI

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
cd whisper_pipeline
pip install -r requirements.txt

# Setup Ollama
ollama serve
ollama pull llama3.2

# Login to Hugging Face
huggingface-cli login
```

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, recommended)
- Ollama (for llama3.2 LLM)
- Hugging Face account (for CSM access)
- ~15GB disk space for models

## Project Structure

```
99StepsAI/
├── whisper_pipeline/          # Conversational chatbot project
│   ├── conversational_chatbot.py   # Main chatbot ⭐
│   ├── conversation_llm.py         # Dual-mode LLM
│   ├── csm_integration.py          # CSM wrapper
│   ├── asr_whisper.py              # Whisper ASR
│   ├── logs/                       # Log files
│   │   └── chatbot.log            # Main log
│   └── README.md                   # Full docs
├── csm/                       # CSM text-to-speech
│   ├── generator.py
│   ├── models.py
│   └── README.md
└── venv/                      # Virtual environment
```

## How It Works

The chatbot uses a unique dual-LLM architecture:

1. **LLM Mode 1**: Generates intelligent, context-aware responses
2. **CSM Processing**: Adds natural conversational audio quality
3. **Whisper**: Transcribes the audio back to text
4. **LLM Mode 2**: Cleans the transcription (removes fillers, preserves meaning)

**Why this approach?**
- Natural conversation from LLM
- Audio tone and quality from CSM
- Clean, professional text output
- Context awareness across dialogue

## Models Used

- **Llama 3.2** (via Ollama) - Conversational AI and text cleanup
- **CSM-1B** - Natural text-to-speech conversion
- **Whisper** - High-quality speech recognition

## Performance

Approximate processing times (on RTX 3090):
- LLM response: ~1-3 seconds
- CSM audio: ~5-10 seconds
- Whisper: ~1-2 seconds
- LLM cleanup: ~1-2 seconds

**Total: ~8-17 seconds per message**

## License

- Whisper: MIT License
- CSM: Apache 2.0 License
- Llama 3.2: Meta Community License

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## Support

For issues:
- Check [whisper_pipeline/README.md](whisper_pipeline/README.md) for detailed docs
- Review logs in `whisper_pipeline/logs/chatbot.log`
- Ensure Ollama is running with llama3.2
- Verify Hugging Face access

---

**Made with ❤️ for AI voice applications**

**Ready to chat?** → `python whisper_pipeline/conversational_chatbot.py` 🚀
