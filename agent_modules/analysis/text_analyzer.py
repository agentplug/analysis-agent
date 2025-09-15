#!/usr/bin/env python3
"""
Text Analyzer Module
Handles text analysis and insights generation.
"""

from typing import Dict, Any
from ..utils.ai_client import AIClientWrapper
from ..utils.response_parser import ResponseParser
from ..utils.config_loader import get_analysis_config


class TextAnalyzer:
    """Handles text analysis and insights generation."""
    
    def __init__(self):
        """Initialize text analyzer."""
        self.ai_client = AIClientWrapper()
        self.parser = ResponseParser()
        self.analysis_config = get_analysis_config()
    
    def analyze_text(self, text: str, analysis_type: str = None, tool_context: str = "") -> Dict[str, Any]:
        """
        Analyze text and provide insights using AI.
        
        Args:
            text: Text content to analyze
            analysis_type: Type of analysis (sentiment, key_points, summary, general). If None, uses config default.
            tool_context: Additional tool context information
            
        Returns:
            Analysis results as a dictionary
        """
        try:
            # Use config default if analysis_type is not provided
            if analysis_type is None:
                analysis_type = self.analysis_config.get('default_analysis_type', 'general')
            # Build system prompt based on analysis type
            system_prompt = self._build_analysis_prompt(analysis_type, tool_context)
            
            # Generate AI response
            response_text = self.ai_client.generate_response(
                system_prompt,
                f"Analyze this text:\n{text}"
            )
            
            # Parse the response
            analysis_result = self.parser.parse_json_response(response_text)
            
            # Add metadata
            analysis_result["analysis_type"] = analysis_type
            analysis_result["text_length"] = len(text)
            analysis_result["status"] = "success"
            
            return analysis_result
            
        except Exception as e:
            return {
                "error": f"Error analyzing text: {str(e)}",
                "analysis_type": analysis_type,
                "status": "error"
            }
    
    def summarize_content(self, content: str, max_length: int = None) -> str:
        """
        Create a summary of content using AI.
        
        Args:
            content: Content to summarize
            max_length: Maximum summary length. If None, uses config default.
            
        Returns:
            Summarized content as a string
        """
        try:
            # Use config default if max_length is not provided
            if max_length is None:
                max_length = self.analysis_config.get('max_summary_length', 200)
            system_prompt = f"""You are a content summarizer. Create a concise summary of the content in approximately {max_length} characters or less. Focus on the main ideas and key information."""
            
            return self.ai_client.generate_response(
                system_prompt,
                f"Summarize this content:\n{content}"
            )
            
        except Exception as e:
            return f"Error summarizing content: {str(e)}. Please check your API key and internet connection."
    
    def _build_analysis_prompt(self, analysis_type: str, tool_context: str = "") -> str:
        """
        Build system prompt for analysis based on type.
        
        Args:
            analysis_type: Type of analysis to perform
            tool_context: Additional tool context
            
        Returns:
            Complete system prompt string
        """
        base_prompts = {
            "sentiment": "You are a sentiment analyzer. Analyze the sentiment of the text and provide insights about emotional tone, positivity/negativity, and key emotional indicators. Return your analysis in JSON format with keys: sentiment, confidence, emotional_tone, key_indicators.",
            "key_points": "You are a key points extractor. Extract and summarize the most important points from the text. Return your analysis in JSON format with keys: main_points, supporting_details, importance_ranking.",
            "summary": "You are a text summarizer. Create a concise summary of the main ideas. Return your analysis in JSON format with keys: summary, main_themes, word_count.",
            "mathematical": "You are a mathematical analysis expert. Analyze mathematical content and provide insights. If calculations are involved and you have tools available, use them. Return your analysis in JSON format with keys: mathematical_content, calculations_needed, methodology, insights.",
            "general": "You are a comprehensive text analyzer. Provide general insights about the text including main themes, tone, structure, and key insights. Return your analysis in JSON format with keys: main_themes, tone, structure, key_insights, summary."
        }
        
        base_prompt = base_prompts.get(analysis_type, base_prompts["general"])
        
        if tool_context:
            return f"{base_prompt}\n\n{tool_context}"
        
        return base_prompt
