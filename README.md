# Analysis Agent

**Version**: 1.0.0  
**Author**: agentplug  
**License**: MIT  

## Description

The Analysis Agent analyzes text content and provides insights. It can perform various types of analysis including sentiment analysis, key point extraction, and content summarization.

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

- `aisuite[openai]>=0.1.7` - AI service integration
- `python-dotenv>=1.0.0` - Environment variable management
- `docstring-parser>=0.17.0` - Required by aisuite

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

3. **Set up API key (optional):**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   # or create ~/.agenthub/.env file
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
