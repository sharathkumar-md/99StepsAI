# Whisper Pipeline - AI Conversational Chatbot 🤖

**Full conversational AI with voice processing pipeline**

```
User Input → CSM → Audio → Whisper → LLM (llama3.2) → Response
```

## Features

- ✅ **Interactive Conversational Chatbot** - Natural dialogue with history
- ✅ **Voice Processing Pipeline** - CSM → Whisper → LLM
- ✅ **Llama 3.2 Integration** - Local LLM via Ollama
- ✅ **Proper Logging** - Professional logging throughout
- ✅ **GPU Acceleration** - CUDA support for faster processing

## Quick Start

### 1. Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional but recommended)
- Ollama installed
- Hugging Face account with Llama access

### 2. Setup

```bash
# Activate virtual environment
cd "C:\Users\Sharath Kumar MD\VSCodE\GitHub\99StepsAI"
venv\Scripts\activate
cd whisper_pipeline

# Install dependencies
pip install -r requirements.txt

# Install Ollama and pull model
ollama serve
ollama pull llama3.2

# Login to Hugging Face
huggingface-cli login
```

**Get Hugging Face Access:**
1. Create account: https://huggingface.co/join
2. Request Llama access: https://huggingface.co/meta-llama/Llama-3.2-1B
3. Get token: https://huggingface.co/settings/tokens

### 3. Run Chatbot

```bash
python conversational_chatbot.py
```

### 4. Chat!

```
👤 You: Hello! How are you?
🤖 Bot: I'm doing great! How can I help you today?

👤 You: Tell me about Python
🤖 Bot: Python is a versatile programming language...

Commands:
  /quit  - Exit chatbot
  /reset - Reset conversation history
```

## Architecture

### Pipeline Flow

```
┌──────────────────────────────────────────────┐
│ User Input (Text or Speech)                  │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ STEP 1: CSM (Conversational Speech Model)   │
│ • Converts text to audio                    │
│ • Normalizes audio output                   │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ STEP 2: Whisper (Speech-to-Text)            │
│ • Transcribes audio to text                 │
│ • Language detection                        │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ STEP 3: LLM (llama3.2)                      │
│ • Generates conversational response         │
│ • Maintains context & history               │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ Response Output                              │
└──────────────────────────────────────────────┘
```

### Components

**Core Modules:**
- `csm_integration.py` - CSM text-to-speech wrapper
- `asr_whisper.py` - Whisper ASR (Automatic Speech Recognition)
- `conversation_llm.py` - LLM conversation manager with history
- `csm_pipeline.py` - Complete pipeline orchestration

**Applications:**
- `conversational_chatbot.py` - **Interactive chatbot** ⭐
- `run_csm_pipeline.py` - Single message processor

## Usage Examples

### Interactive Chatbot (Recommended)

```bash
python conversational_chatbot.py
```

Natural conversation with maintained history.

### Single Message Mode

```bash
# Text input
python run_csm_pipeline.py "How's the weather today?" --text

# Audio file input
python run_csm_pipeline.py audio.wav --audio
```

### Python API

```python
from csm_pipeline import CSMPipeline

# Initialize pipeline
pipeline = CSMPipeline(
    whisper_model_size="base",
    llm_model="llama3.2"
)

# Process text
result = pipeline.process_text("Hello, how are you?")

# Process audio file
result = pipeline.process_audio("audio.wav")

# Auto-detect input type
csm_audio, whisper_text, response = pipeline.process(input_data)
```

### Conversational LLM

```python
from conversation_llm import ConversationLLM

llm = ConversationLLM(llm_model="llama3.2")

# Chat with history
response = llm.chat("Hello!")
response = llm.chat("What did I just say?")  # Remembers context

# Reset conversation
llm.reset()
```

## Configuration

### Model Selection

```python
# Whisper model sizes
# tiny    - Fastest, least accurate (~1GB VRAM)
# base    - Good balance (recommended, ~1GB VRAM)
# small   - Better accuracy (~2GB VRAM)
# medium  - High accuracy (~5GB VRAM)
# large   - Best accuracy (~10GB VRAM)

pipeline = CSMPipeline(
    whisper_model_size="base",
    llm_model="llama3.2",
    csm_device=None,  # 'cuda', 'cpu', or None (auto-detect)
    language="en"
)
```

### Environment Variables

Create `.env` file (optional):
```bash
OLLAMA_HOST=http://localhost:11434
```

## Requirements

### System Requirements
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- CUDA-compatible GPU (optional, recommended)
- ~15GB disk space (for models)

### Software Dependencies
- PyTorch with CUDA support
- OpenAI Whisper
- Ollama (for LLM)
- CSM (Conversational Speech Model)
- See `requirements.txt` for full list

### External Services
- Ollama running locally
- Hugging Face account (for CSM model access)

## Troubleshooting

### venv Not Activated
```bash
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### Ollama Not Running
```bash
ollama serve
ollama pull llama3.2
ollama list  # Verify llama3.2 is installed
```

### CSM Model Not Found
```bash
# Login to Hugging Face
huggingface-cli login

# Verify CSM installation in parent directory
ls ../csm/
```

### CUDA Out of Memory
- Use smaller Whisper model: `whisper_model_size="tiny"` or `"base"`
- Reduce batch sizes
- Close other GPU applications

### Import Errors
```bash
pip install -r requirements.txt --upgrade
```

## Logging

All components use Python's logging module:

```python
import logging

# Set log level
logging.basicConfig(level=logging.INFO)  # INFO, DEBUG, WARNING, ERROR
```

Logs show:
- INFO: Normal progress updates
- WARNING: Non-critical issues
- ERROR: Failures and exceptions

## Project Structure

```
whisper_pipeline/
├── conversational_chatbot.py   # Interactive chatbot (main)
├── conversation_llm.py          # LLM with conversation history
├── csm_integration.py           # CSM wrapper
├── asr_whisper.py               # Whisper ASR
├── csm_pipeline.py              # Pipeline orchestration
├── run_csm_pipeline.py          # CLI runner
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## Models Used

- **CSM-1B**: Conversational Speech Model for text-to-speech
- **Whisper**: OpenAI's speech recognition model
- **Llama 3.2**: Meta's LLM via Ollama for conversation

## Performance

Approximate processing times (on RTX 3090):
- CSM audio generation: ~5-10 seconds per message
- Whisper transcription: ~1-2 seconds per 10 seconds of audio
- LLM response: ~1-3 seconds per message

## License

This project uses:
- CSM: Apache 2.0 License
- Whisper: MIT License
- Llama 3.2: Meta Community License

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## Support

For issues and questions:
- Check troubleshooting section above
- Review logs for error details
- Ensure all dependencies are installed
- Verify Ollama is running with llama3.2

## Notes

- **Always activate venv before running!**
- CSM requires Hugging Face access (one-time setup)
- GPU acceleration requires CUDA-compatible hardware
- First run downloads models (~10GB total)

---

**Ready to chat? Run `python conversational_chatbot.py` and start talking!** 🚀
