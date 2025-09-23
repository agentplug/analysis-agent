# Analysis Agent

**Version**: 1.0.0  
**Author**: agentplug  
**License**: MIT  

## Description

The Analysis Agent analyzes text content and provides insights using an enhanced LLM service inspired by agenthub core. It features automatic model detection, local model support (Ollama), intelligent model scoring, and comprehensive error handling. The agent can perform various types of analysis including sentiment analysis, key point extraction, and content summarization.

## Enhanced LLM Features

### Automatic Model Detection
- **Local Models**: Automatically detects and uses Ollama models when available
- **Cloud Models**: Falls back to cloud providers based on available API keys
- **Intelligent Scoring**: Selects the best model based on size, family, and quality scores
- **Smart Fallbacks**: Graceful degradation when models are unavailable

### Supported Models
- **Local**: Ollama models (llama3, gemma, qwen, mistral, etc.)
- **Cloud**: OpenAI, Anthropic, Google, DeepSeek, Groq, and more
- **Auto-Detection**: Automatically finds the best available model

### Advanced Capabilities
- **JSON Responses**: Structured output for reliable parsing
- **Shared Instances**: Prevents duplicate model detection logs
- **Comprehensive Logging**: Detailed information about model selection
- **Error Handling**: Robust fallback mechanisms

## Methods

### `analyze_text(text: str, analysis_type: str = "general") -> dict`

Analyzes text and provides insights based on the specified analysis type.

**Parameters:**
- `text` (string, required): Text content to analyze
- `analysis_type` (string, optional): Type of analysis - "sentiment", "key_points", "summary", or "general" (default)

**Returns:**
- Analysis results as a dictionary with insights

**Example:**
```bash
python agent.py '{"method": "analyze_text", "parameters": {"text": "Python is a great programming language.", "analysis_type": "sentiment"}}'
```

### `summarize_content(content: str, max_length: int = 200) -> str`

Creates a summary of the provided content.

**Parameters:**
- `content` (string, required): Content to summarize
- `max_length` (integer, optional): Maximum summary length in characters (default: 200)

**Returns:**
- Summarized content as a string

**Example:**
```bash
python agent.py '{"method": "summarize_content", "parameters": {"content": "Long text content here...", "max_length": 150}}'
```

## Analysis Types

### General Analysis
Provides comprehensive insights including main themes, tone, structure, and key insights.

### Sentiment Analysis
Analyzes emotional tone, positivity/negativity, and emotional indicators.

### Key Points
Extracts and ranks the most important points from the text.

### Summary
Creates a concise summary of the main ideas and themes.

## Dependencies

- `aisuite[openai]>=0.1.11` - AI service integration with multi-provider support
- `python-dotenv>=1.1.1` - Environment variable management
- `docstring-parser>=0.17.0` - Required by aisuite
- `mcp>=1.14.0` - Model Context Protocol support
- `requests>=2.31.0` - HTTP requests for Ollama detection

## Setup

1. **Create virtual environment:**
   ```bash
   uv venv .venv
   source .venv/bin/activate  # Unix/macOS
   # or .venv\Scripts\activate  # Windows
   ```

2. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

3. **Set up API keys (optional):**
   ```bash
   # For cloud models (choose one or more)
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   export GOOGLE_API_KEY="your-google-key"
   export DEEPSEEK_API_KEY="your-deepseek-key"
   export GROQ_API_KEY="your-groq-key"
   
   # For local Ollama models
   # Install Ollama and pull models: ollama pull llama3
   export OLLAMA_API_URL="http://localhost:11434"  # Optional, auto-detected
   ```

## Usage

The agent accepts JSON input via command line and returns JSON output:

```bash
# Activate virtual environment first
source .venv/bin/activate

# General text analysis
python agent.py '{"method": "analyze_text", "parameters": {"text": "Your text here"}}'

# Sentiment analysis
python agent.py '{"method": "analyze_text", "parameters": {"text": "I love this product!", "analysis_type": "sentiment"}}'

# Content summarization
python agent.py '{"method": "summarize_content", "parameters": {"content": "Long article content..."}}'
```

## Enhanced Features

The agent now includes:
- Automatic model detection and selection
- Available models listing
- Enhanced text analysis capabilities
- Shared client management
- JSON response generation

## Model Detection Priority

The agent automatically selects models in this order:

1. **Local Ollama Models** (highest priority)
   - Detects running Ollama instances
   - Scores models by size and family
   - Prefers larger, higher-quality models

2. **Cloud Models** (fallback)
   - Based on available API keys
   - Supports multiple providers simultaneously

3. **Default Fallback**
   - Uses OpenAI GPT-4o-mini if no other models available

## Error Handling

The agent gracefully handles:
- Missing API keys (provides fallback responses)
- Invalid method names
- Missing parameters
- Network connectivity issues
- AI service errors
- Invalid JSON responses from AI

All errors are returned in JSON format with an `error` field.

## Tags

- text-analysis
- insights
- ai-assistant
