#!/bin/bash
# Quick test script for streaming CSM on SageMaker

echo "=================================================="
echo "STREAMING CSM SETUP & TEST"
echo "=================================================="

# Install dependencies
echo "📦 Installing faster-whisper..."
pip install faster-whisper

# Test streaming chatbot
echo ""
echo "🚀 Testing streaming chatbot..."
python streaming_chatbot.py

echo ""
echo "=================================================="
echo "✅ Test complete! Check logs above for performance metrics."
echo "=================================================="
