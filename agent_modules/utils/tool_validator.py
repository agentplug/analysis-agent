#!/usr/bin/env python3
"""
Tool Validator Module
Handles validation of tool calls and authorization checks.
"""

import sys
from typing import Dict, Any, List

# Custom print function to avoid JSON parsing issues
def log_print(*args, **kwargs):
    """Print to stderr to avoid interfering with JSON output"""
    print(*args, file=sys.stderr, **kwargs)


class ToolValidator:
    """Validates tool calls and ensures authorization."""
    
    def __init__(self, available_tools: List[str]):
        """
        Initialize tool validator.
        
        Args:
            available_tools: List of authorized tool names
        """
        self.available_tools = available_tools
    
    def validate_tool_call(self, tool_call: Dict[str, Any]) -> bool:
        """
        Validate tool call structure and ensure tool is in assigned tools list.
        
        Args:
            tool_call: Tool call dictionary to validate
            
        Returns:
            True if tool call is valid and allowed, False otherwise
        """
        if not isinstance(tool_call, dict):
            log_print(f"❌ Invalid tool call: not a dictionary")
            return False
        
        if "tool_name" not in tool_call:
            log_print(f"❌ Invalid tool call: missing 'tool_name'")
            return False
        
        if not isinstance(tool_call["tool_name"], str):
            log_print(f"❌ Invalid tool call: 'tool_name' must be string")
            return False
        
        tool_name = tool_call["tool_name"]
        
        # CRITICAL: Ensure tool is in the assigned tools list
        if tool_name not in self.available_tools:
            log_print(f"❌ UNAUTHORIZED TOOL ACCESS: '{tool_name}' is not in assigned tools: {self.available_tools}")
            return False
        
        # Validate arguments structure
        if "arguments" in tool_call:
            if not isinstance(tool_call["arguments"], dict):
                log_print(f"❌ Invalid tool call: 'arguments' must be dictionary")
                return False
        
        log_print(f"✅ Tool call validated: '{tool_name}' is authorized")
        return True
    
    def is_tool_authorized(self, tool_name: str) -> bool:
        """
        Check if a tool is authorized for use.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if authorized, False otherwise
        """
        return tool_name in self.available_tools
    
    def get_unauthorized_tools(self, tool_calls: List[Dict[str, Any]]) -> List[str]:
        """
        Get list of unauthorized tools from a list of tool calls.
        
        Args:
            tool_calls: List of tool call dictionaries
            
        Returns:
            List of unauthorized tool names
        """
        unauthorized = []
        for tool_call in tool_calls:
            if isinstance(tool_call, dict) and "tool_name" in tool_call:
                tool_name = tool_call["tool_name"]
                if not self.is_tool_authorized(tool_name):
                    unauthorized.append(tool_name)
        return unauthorized
    
    def validate_tool_calls_batch(self, tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate a batch of tool calls.
        
        Args:
            tool_calls: List of tool call dictionaries
            
        Returns:
            Validation result with valid and invalid tool calls
        """
        valid_calls = []
        invalid_calls = []
        
        for tool_call in tool_calls:
            if self.validate_tool_call(tool_call):
                valid_calls.append(tool_call)
            else:
                invalid_calls.append(tool_call)
        
        return {
            "valid_calls": valid_calls,
            "invalid_calls": invalid_calls,
            "total_calls": len(tool_calls),
            "valid_count": len(valid_calls),
            "invalid_count": len(invalid_calls),
            "all_valid": len(invalid_calls) == 0
        }
