#!/usr/bin/env python3
"""
Example script demonstrating how to use the configuration system.
"""

from agent_modules.utils.ai_client import AIClientWrapper
from agent_modules.utils.config_loader import get_ai_config, get_analysis_config, get_config_loader
from agent_modules.analysis.text_analyzer import TextAnalyzer


def main():
    """Demonstrate configuration usage."""
    
    print("=== Configuration System Example ===\n")
    
    # 1. Show current AI configuration
    print("1. Current AI Configuration:")
    ai_config = get_ai_config()
    for key, value in ai_config.items():
        print(f"   {key}: {value}")
    print()
    
    # 2. Show current analysis configuration
    print("2. Current Analysis Configuration:")
    analysis_config = get_analysis_config()
    for key, value in analysis_config.items():
        print(f"   {key}: {value}")
    print()
    
    # 3. Create AI client with default config
    print("3. Creating AI client with default configuration:")
    ai_client = AIClientWrapper()
    print(f"   Model: {ai_client.model}")
    print(f"   Temperature: {ai_client.temperature}")
    print(f"   Max Tokens: {ai_client.max_tokens}")
    print(f"   Top P: {ai_client.top_p}")
    print()
    
    # 4. Create AI client with custom parameters (overrides config)
    print("4. Creating AI client with custom parameters:")
    custom_client = AIClientWrapper(
        model="openai:gpt-4",
        temperature=0.7,
        max_tokens=1000
    )
    print(f"   Model: {custom_client.model}")
    print(f"   Temperature: {custom_client.temperature}")
    print(f"   Max Tokens: {custom_client.max_tokens}")
    print()
    
    # 5. Show how to get specific config values
    print("5. Getting specific configuration values:")
    config_loader = get_config_loader()
    model = config_loader.get('ai.model')
    default_analysis = config_loader.get('analysis.default_analysis_type')
    max_summary = config_loader.get('analysis.max_summary_length')
    print(f"   AI Model: {model}")
    print(f"   Default Analysis Type: {default_analysis}")
    print(f"   Max Summary Length: {max_summary}")
    print()
    
    # 6. Demonstrate TextAnalyzer using config
    print("6. TextAnalyzer using configuration:")
    analyzer = TextAnalyzer()
    print(f"   AI Client Model: {analyzer.ai_client.model}")
    print(f"   Default Analysis Type: {analyzer.analysis_config.get('default_analysis_type')}")
    print(f"   Max Summary Length: {analyzer.analysis_config.get('max_summary_length')}")
    print()
    
    print("=== Configuration Example Complete ===")


if __name__ == "__main__":
    main()
