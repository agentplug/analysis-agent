#!/usr/bin/env python3
"""
Response Parser Module
Handles parsing of AI responses to extract tool calls and structured data.
"""

import json
import re
from typing import List, Dict, Any


class ResponseParser:
    """Parses AI responses to extract tool calls and structured information."""
    
    def __init__(self):
        """Initialize response parser."""
        pass
    
    def extract_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from AI response with detailed logging.
        
        Args:
            response: AI response string to parse
            
        Returns:
            List of tool call dictionaries
        """
        print("🔍 AGENT TOOL SELECTION: Analyzing AI response for tool calls...")
        print(f"   📝 Response length: {len(response)} characters")
        
        tool_calls = []
        
        # First, try to parse the entire response as JSON
        try:
            full_response = json.loads(response)
            if "tool_call" in full_response:
                tool_call = full_response["tool_call"]
                print(f"   ✅ Found tool call in JSON: {tool_call}")
                tool_calls.append(tool_call)
                return tool_calls
        except json.JSONDecodeError:
            print("   📋 Response is not pure JSON, searching for embedded tool calls...")
        
        # Check for JSON wrapped in markdown code blocks
        json_pattern = r'```json\s*(\{.*?\})\s*```'
        json_matches = re.findall(json_pattern, response, re.DOTALL)
        for json_match in json_matches:
            try:
                json_obj = json.loads(json_match)
                if "tool_call" in json_obj:
                    tool_call = json_obj["tool_call"]
                    print(f"   ✅ Found tool call in markdown JSON: {tool_call}")
                    tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue
        
        # Look for JSON objects containing tool_call in the response
        # Use a more flexible approach to find JSON objects
        lines = response.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if '"tool_call"' in line:
                print(f"   🔍 Found 'tool_call' in line {i+1}: {line[:100]}...")
                try:
                    # Try to parse the line as JSON
                    obj = json.loads(line)
                    if "tool_call" in obj:
                        tool_call = obj["tool_call"]
                        print(f"   ✅ Extracted tool call: {tool_call}")
                        tool_calls.append(tool_call)
                except json.JSONDecodeError:
                    # If the line isn't complete JSON, try to find JSON within it
                    json_pattern = r'\{[^{}]*"tool_call"[^{}]*\}'
                    matches = re.findall(json_pattern, line)
                    for match in matches:
                        try:
                            tool_call_obj = json.loads(match)
                            if "tool_call" in tool_call_obj:
                                tool_call = tool_call_obj["tool_call"]
                                print(f"   ✅ Extracted tool call from pattern: {tool_call}")
                                tool_calls.append(tool_call)
                        except json.JSONDecodeError:
                            continue
        
        print(f"🎯 TOOL SELECTION RESULT: Found {len(tool_calls)} tool calls")
        for i, tool_call in enumerate(tool_calls, 1):
            print(f"   {i}. Tool: {tool_call.get('tool_name', 'unknown')}")
            print(f"      Args: {tool_call.get('arguments', {})}")
        
        return tool_calls
    
    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON response with fallback handling.
        
        Args:
            response: AI response string
            
        Returns:
            Parsed JSON dictionary or error info
        """
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            json_pattern = r'```json\s*(\{.*?\})\s*```'
            json_match = re.search(json_pattern, response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Try to find any JSON-like structure
            json_pattern = r'\{[^{}]*\}'
            json_matches = re.findall(json_pattern, response, re.DOTALL)
            for match in json_matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
            
            # Return raw response if no JSON found
            return {"raw_response": response, "parse_error": "Could not extract JSON"}
    
    def extract_step_summary(self, result: Dict[str, Any]) -> str:
        """Extract a brief summary from step execution result."""
        if isinstance(result, dict):
            if "summary" in result:
                return result["summary"]
            elif "result" in result:
                summary = str(result["result"])
                return summary[:200] + "..." if len(summary) > 200 else summary
            elif "analysis" in result:
                return result["analysis"]
            else:
                return "Step completed successfully"
        else:
            summary = str(result)
            return summary[:200] + "..." if len(summary) > 200 else summary
    
    def is_completion_response(self, response: str) -> bool:
        """
        Check if response indicates completion.
        
        Args:
            response: AI response string
            
        Returns:
            True if response indicates completion
        """
        completion_indicators = [
            "COMPLETE",
            "complete",
            "finished",
            "done",
            "no more steps",
            "task complete"
        ]
        
        response_lower = response.lower().strip()
        return any(indicator in response_lower for indicator in completion_indicators)
