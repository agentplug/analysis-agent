# Configuration System

The analysis agent now supports configuration through a `config.json` file, allowing you to customize AI model selection, parameters, and other settings without modifying code.

## Configuration File

The `config.json` file is located in the project root and contains the following sections:

### AI Configuration (`ai` section)

- `model`: AI model identifier (e.g., "openai:gpt-4o-mini", "openai:gpt-4")
- `temperature`: Response randomness (0.0 to 1.0)
- `max_tokens`: Maximum tokens in response (null for no limit)
- `top_p`: Nucleus sampling parameter (0.0 to 1.0)
- `frequency_penalty`: Frequency penalty (-2.0 to 2.0)
- `presence_penalty`: Presence penalty (-2.0 to 2.0)

### Analysis Configuration (`analysis` section)

- `default_analysis_type`: Default analysis type ("general", "sentiment", "key_points", "summary", "mathematical")
- `max_summary_length`: Default maximum summary length in characters
- `enable_streaming`: Whether to enable streaming responses

### Execution Configuration (`execution` section)

- `timeout_seconds`: Timeout for tool execution
- `max_retries`: Maximum retry attempts
- `retry_delay`: Delay between retries in seconds

### Logging Configuration (`logging` section)

- `level`: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
- `format`: Log message format

## Usage Examples

### Basic Usage

```python
from agent_modules.utils.ai_client import AIClientWrapper

# Uses configuration from config.json
ai_client = AIClientWrapper()

# Override specific parameters
ai_client = AIClientWrapper(
    model="openai:gpt-4",
    temperature=0.7
)
```

### Accessing Configuration

```python
from agent_modules.utils.config_loader import get_ai_config, get_analysis_config

# Get AI configuration
ai_config = get_ai_config()
print(f"Model: {ai_config['model']}")

# Get analysis configuration
analysis_config = get_analysis_config()
print(f"Default analysis type: {analysis_config['default_analysis_type']}")
```

### Using with TextAnalyzer

```python
from agent_modules.analysis.text_analyzer import TextAnalyzer

# Uses configuration defaults
analyzer = TextAnalyzer()

# Analysis will use default_analysis_type from config
result = analyzer.analyze_text("Your text here")

# Summary will use max_summary_length from config
summary = analyzer.summarize_content("Your content here")
```

## Configuration Loading

The configuration is automatically loaded when modules are imported. You can also manually reload configuration:

```python
from agent_modules.utils.config_loader import get_config_loader

config_loader = get_config_loader()
config_loader.reload_config()  # Reload from file
```

## Custom Configuration Path

You can specify a custom configuration file path:

```python
from agent_modules.utils.config_loader import ConfigLoader

config_loader = ConfigLoader("/path/to/your/config.json")
```

## Example Configuration

```json
{
  "ai": {
    "model": "openai:gpt-4o-mini",
    "temperature": 0.0,
    "max_tokens": null,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
  },
  "analysis": {
    "default_analysis_type": "general",
    "max_summary_length": 200,
    "enable_streaming": false
  },
  "execution": {
    "timeout_seconds": 30,
    "max_retries": 3,
    "retry_delay": 1.0
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  }
}
```

## Benefits

1. **Easy Model Switching**: Change AI models without code modifications
2. **Parameter Tuning**: Adjust temperature, tokens, and other parameters
3. **Environment-Specific Settings**: Different configs for development/production
4. **Centralized Configuration**: All settings in one place
5. **Backward Compatibility**: Existing code continues to work with sensible defaults
