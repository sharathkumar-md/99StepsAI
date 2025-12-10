# Whisper Pipeline - AI Conversational Chatbot 🤖

**Conversational AI with Dual-LLM + Voice Processing Pipeline**

```
User Input → LLM Response → CSM Audio → Whisper → LLM Cleanup → Clean Output
```

## Features

- ✅ **Dual-Mode LLM** - Response generation + text cleanup (llama3.2)
- ✅ **Natural Conversation** - Full dialogue with context and history
- ✅ **CSM Audio Quality** - Natural conversational speech synthesis
- ✅ **Clean Text Output** - Removes fillers while preserving meaning
- ✅ **GPU Acceleration** - CUDA support for faster processing
- ✅ **Professional Logging** - All logs saved to `logs/chatbot.log`

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
🤖 Bot: Hi! I'm doing great, thanks for asking. How can I help you today?

👤 You: Tell me about Python
🤖 Bot: Python is a versatile programming language that's great for beginners...

Commands:
  /quit  - Exit chatbot
  /reset - Reset conversation history
```

## Architecture

### Complete Pipeline Flow

```
┌──────────────────────────────────────────────┐
│ 1. User Input (Text or Audio)               │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ 2. LLM Response Generation (llama3.2)       │
│ • Generates conversational response         │
│ • Maintains conversation history            │
│ • Context-aware dialogue                    │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ 3. CSM Text-to-Speech                       │
│ • Converts LLM response to audio            │
│ • Natural conversational quality            │
│ • Emotional tone and intonation             │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ 4. Whisper Speech-to-Text                   │
│ • Transcribes CSM audio                     │
│ • May include fillers (um, uh, etc.)        │
│ • High-quality transcription                │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ 5. LLM Text Cleanup (llama3.2)              │
│ • Removes filler words                      │
│ • Fixes transcription errors                │
│ • Preserves original meaning                │
└──────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────┐
│ Clean Response Output ✨                     │
└──────────────────────────────────────────────┘
```

### Why This Architecture?

**Dual-LLM Design:**
- **LLM Mode 1**: Generates intelligent, context-aware responses
- **CSM Processing**: Adds natural conversational audio quality
- **LLM Mode 2**: Cleans transcription while preserving meaning

**Benefits:**
- Natural conversational flow from LLM
- Audio quality and tone from CSM
- Clean, professional text output
- Context awareness across turns

## Components

### Core Modules (4 Files)

```
whisper_pipeline/
├── conversational_chatbot.py   # Main chatbot application ⭐
├── conversation_llm.py          # Dual-mode LLM manager
├── csm_integration.py           # CSM text-to-speech wrapper
└── asr_whisper.py               # Whisper speech recognition
```

**1. conversational_chatbot.py**
- Interactive chatbot interface
- Complete pipeline orchestration
- Chat commands (/quit, /reset)

**2. conversation_llm.py**
- **Mode 1**: Response generation with history
- **Mode 2**: Text cleanup without history
- Uses llama3.2 via Ollama

**3. csm_integration.py**
- CSM text-to-speech conversion
- Natural conversational audio
- Temporary file management

**4. asr_whisper.py**
- Whisper speech-to-text
- Multiple model sizes
- Language detection

## Usage

### Interactive Chatbot

```bash
python conversational_chatbot.py
```

Full conversational experience with:
- Context-aware responses
- Conversation history
- Natural dialogue flow
- Clean text output

### Python API

```python
from conversational_chatbot import ConversationalChatbot

# Initialize chatbot
chatbot = ConversationalChatbot(
    whisper_model_size="base",
    llm_model="llama3.2"
)

# Chat (text input)
response = chatbot.chat("Hello! How are you?")
print(response)  # Clean conversational response

# Process audio input
response = chatbot.process_user_input("audio.wav", input_type="audio")

# Reset conversation
chatbot.reset()
```

### Dual-Mode LLM

```python
from conversation_llm import ConversationLLM

llm = ConversationLLM(llm_model="llama3.2")

# Mode 1: Response Generation (with history)
response = llm.generate_response("What's the weather like?")
# Output: "I don't have access to real-time weather data..."

# Mode 2: Text Cleanup (no history)
dirty = "Um, hello there, uh, how are you?"
clean = llm.cleanup_text(dirty)
# Output: "Hello there, how are you?"
```

## Configuration

### Model Selection

```python
chatbot = ConversationalChatbot(
    whisper_model_size="base",  # tiny, base, small, medium, large
    llm_model="llama3.2",       # Always use llama3.2
    csm_device=None             # 'cuda', 'cpu', or None (auto)
)
```

**Whisper Model Sizes:**
- `tiny` - Fastest, least accurate (~1GB VRAM)
- `base` - Good balance (recommended, ~1GB VRAM)
- `small` - Better accuracy (~2GB VRAM)
- `medium` - High accuracy (~5GB VRAM)
- `large` - Best accuracy (~10GB VRAM)

### Environment Variables

Create `.env` file (optional):
```bash
OLLAMA_HOST=http://localhost:11434
```

## Logging

All activity is logged to **`logs/chatbot.log`**

Logs include:
- User inputs
- LLM responses (both modes)
- CSM audio generation
- Whisper transcriptions
- Final clean outputs
- Errors and warnings

View logs:
```bash
tail -f logs/chatbot.log
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
- Ollama (for llama3.2)
- CSM (Conversational Speech Model)
- See `requirements.txt` for full list

### External Services
- **Ollama** running locally with llama3.2
- **Hugging Face** account (for CSM model access)

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

### CSM Model Access Error
```bash
# Login to Hugging Face
huggingface-cli login

# Request access to Llama-3.2-1B
# Visit: https://huggingface.co/meta-llama/Llama-3.2-1B
```

### CUDA Out of Memory
- Use smaller Whisper model: `whisper_model_size="tiny"` or `"base"`
- Close other GPU applications
- Reduce batch sizes

### Import Errors
```bash
pip install -r requirements.txt --upgrade
```

### Warnings Flooding Output
All torch/triton warnings are suppressed. If you still see warnings:
```python
import warnings
warnings.filterwarnings('ignore')
```

## How It Works

### Step-by-Step Example

**Input:**
```
You: "Hello, what's your name?"
```

**Step 1 - LLM Response Generation:**
```
LLM generates: "Hi! I'm your AI assistant. How can I help you today?"
```

**Step 2 - CSM Audio:**
```
CSM converts to natural conversational audio (with tone, emotion)
```

**Step 3 - Whisper Transcription:**
```
Whisper transcribes: "Hi, um, I'm your AI assistant. How can I help you today?"
(May add fillers from audio processing)
```

**Step 4 - LLM Cleanup:**
```
LLM cleans: "Hi! I'm your AI assistant. How can I help you today?"
(Removes fillers, preserves meaning)
```

**Final Output:**
```
Bot: "Hi! I'm your AI assistant. How can I help you today?"
```

### Why Route Through CSM → Whisper?

**Benefits:**
1. **Audio Quality**: CSM adds natural conversational tone
2. **Consistency**: All responses processed uniformly
3. **Realism**: Captures speech patterns and intonation
4. **Cleanup**: LLM removes artifacts while preserving quality

## Models Used

- **Llama 3.2** (via Ollama): Conversational AI and text cleanup
- **CSM-1B**: Natural text-to-speech conversion
- **Whisper**: High-quality speech recognition

## Performance

Approximate processing times (on RTX 3090):
- LLM response generation: ~1-3 seconds
- CSM audio generation: ~5-10 seconds
- Whisper transcription: ~1-2 seconds per 10s audio
- LLM text cleanup: ~1-2 seconds

**Total per message: ~8-17 seconds**

## Project Structure

```
whisper_pipeline/
├── conversational_chatbot.py   # Main chatbot
├── conversation_llm.py          # Dual-mode LLM
├── csm_integration.py           # CSM wrapper
├── asr_whisper.py               # Whisper ASR
├── requirements.txt             # Dependencies
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
├── logs/                        # Log files
│   └── chatbot.log              # Main log file
└── README.md                    # This file
```

## License

This project uses:
- CSM: Apache 2.0 License
- Whisper: MIT License
- Llama 3.2: Meta Community License

## Notes

- **Always activate venv before running!**
- CSM requires Hugging Face access (one-time setup)
- GPU acceleration requires CUDA-compatible hardware
- First run downloads models (~10GB total)
- All logs saved to `logs/chatbot.log`

## Support

For issues and questions:
- Check troubleshooting section above
- Review `logs/chatbot.log` for error details
- Ensure all dependencies are installed
- Verify Ollama is running with llama3.2

---

**Ready to chat? Run `python conversational_chatbot.py` and start talking!** 🚀
