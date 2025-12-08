# 99StepsAI - Voice AI Projects

Collection of AI projects focused on conversational AI and voice processing.

## Projects

### Whisper Pipeline - Conversational Chatbot 🤖

Full-featured AI chatbot with voice processing pipeline.

**Pipeline:** `User Input → CSM → Audio → Whisper → LLM (llama3.2) → Response`

[View Documentation →](whisper_pipeline/README.md)

**Quick Start:**
```bash
cd whisper_pipeline
python conversational_chatbot.py
```

### CSM (Conversational Speech Model)

Text-to-speech generation using CSM from Sesame AI Labs.

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

# Install dependencies for specific project
cd whisper_pipeline
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, recommended)
- Ollama (for LLM)
- Hugging Face account

## Projects Structure

```
99StepsAI/
├── whisper_pipeline/        # Conversational chatbot project
│   ├── conversational_chatbot.py
│   ├── csm_integration.py
│   ├── asr_whisper.py
│   └── ...
├── csm/                     # CSM text-to-speech
│   ├── generator.py
│   ├── models.py
│   └── ...
└── venv/                    # Virtual environment
```

## License

- Whisper: MIT License
- CSM: Apache 2.0 License
- Llama 3.2: Meta Community License

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Submit pull request

---

**Made with ❤️ for AI voice applications**
