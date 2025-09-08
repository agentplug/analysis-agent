#!/usr/bin/env python3
"""
Base Agent Module
Core functionality for the analysis agent.
"""

from typing import Dict, Any, List, Optional
import json


class BaseAgent:
    """Base class for analysis agents with tool integration."""
    
    def __init__(self, tool_context: Optional[Dict[str, Any]] = None):
        """
        Initialize the base agent.
        
        Args:
            tool_context: Dictionary containing tool metadata and context information
        """
        self.tool_context = tool_context or {}
        self.available_tools = self.tool_context.get("available_tools", [])
        self.tool_descriptions = self.tool_context.get("tool_descriptions", {})
        self.tool_usage_examples = self.tool_context.get("tool_usage_examples", {})
        
        # Initialize modules (will be set by concrete implementations)
        self.problem_decomposer = None
        self.tool_executor = None
        self.text_analyzer = None
        self.step_planner = None
    
    def validate_tool_context(self) -> bool:
        """
        Validate tool context structure.
        
        Returns:
            True if tool context is valid, False otherwise
        """
        if not isinstance(self.tool_context, dict):
            return False
        
        # Check if available_tools is a list
        if "available_tools" in self.tool_context:
            if not isinstance(self.tool_context["available_tools"], list):
                return False
        
        # Check if tool_descriptions is a dict
        if "tool_descriptions" in self.tool_context:
            if not isinstance(self.tool_context["tool_descriptions"], dict):
                return False
        
        # Check if tool_usage_examples is a dict
        if "tool_usage_examples" in self.tool_context:
            if not isinstance(self.tool_context["tool_usage_examples"], dict):
                return False
        
        return True
    
    def build_tool_context_string(self) -> str:
        """
        Build tool context string for AI system prompt.
        
        Returns:
            Formatted tool context string
        """
        if not self.available_tools:
            return ""
        
        tool_descriptions = []
        for tool_name in self.available_tools:
            description = self.tool_descriptions.get(tool_name, f"Tool: {tool_name}")
            examples = self.tool_usage_examples.get(tool_name, [])
            
            tool_descriptions.append(f"""
Tool: {tool_name}
Description: {description}
Examples: {', '.join(examples)}
""")
        
        return f"""
You have access to the following tools. CRITICAL RULE: You MUST use the tools for ALL mathematical operations that match your available tools. Do NOT calculate anything manually.

{''.join(tool_descriptions)}

MANDATORY RULES:
1. If text asks for multiplication and you have "multiply" tool → USE IT
2. If text asks for addition and you have "add" tool → USE IT  
3. If text has multiple operations → PERFORM THEM STEP BY STEP
4. NEVER calculate manually if you have the corresponding tool
5. Always use tools for ALL calculations, even simple ones

For multi-step calculations like "multiply 12 by 5, then add 8":
- FIRST: Use multiply tool for 12 × 5  
- The system will execute this and give you the result
- THEN: You'll be asked to continue with the next step

To use a tool, respond with a JSON object containing:
{{
    "tool_call": {{
        "tool_name": "tool_name",
        "arguments": {{"param1": "value1", "param2": "value2"}}
    }},
    "analysis": "I will use the tool to perform this calculation"
}}

EXAMPLES:
- "Calculate 6 times 7" → {{"tool_call": {{"tool_name": "multiply", "arguments": {{"a": "6", "b": "7"}}}}}}
- "Add 5 and 3" → {{"tool_call": {{"tool_name": "add", "arguments": {{"a": "5", "b": "3"}}}}}}  
- "Calculate 12 times 5, then add 8" → {{"tool_call": {{"tool_name": "multiply", "arguments": {{"a": "12", "b": "5"}}}}}}

IMPORTANT: For text like "Calculate 12 times 5, then add 8", you should:
1. First respond with multiply tool call
2. When you get the result (60), the system will continue the conversation
3. Then you'll use the add tool to add 8 to that result

REMEMBER: If you can use a tool, you MUST use it. Do not calculate manually.
"""
    
    def get_tool_info(self) -> Dict[str, Any]:
        """
        Get information about available tools.
        
        Returns:
            Dictionary with tool information
        """
        return {
            "available_tools": self.available_tools,
            "tool_descriptions": self.tool_descriptions,
            "tool_usage_examples": self.tool_usage_examples,
            "tool_count": len(self.available_tools)
        }
    
    def log_execution_step(self, step: str, details: str = "") -> None:
        """
        Log execution step for debugging and transparency.
        
        Args:
            step: Step description
            details: Additional details
        """
        print(f"🔍 AGENT STEP: {step}")
        if details:
            print(f"   📝 Details: {details}")
