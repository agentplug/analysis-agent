#!/usr/bin/env python3
"""
Agent Hub Agent: analysis-agent
Analyzes text content and provides insights.
"""

import json
import sys
import re
from typing import Dict, Any, List, Optional

class AnalysisAgent:
    """Text analysis and insights agent."""
    
    def __init__(self, tool_context: Optional[Dict[str, Any]] = None):
        """
        Initialize the analysis agent with optional tool context.
        
        Args:
            tool_context: Dictionary containing tool metadata and context information
        """
        self.tool_context = tool_context or {}
        self.available_tools = self.tool_context.get("available_tools", [])
        self.tool_descriptions = self.tool_context.get("tool_descriptions", {})
        self.tool_usage_examples = self.tool_context.get("tool_usage_examples", {})
    
    def _build_system_prompt(self, analysis_type: str) -> str:
        """
        Build system prompt with tool context if available.
        
        Args:
            analysis_type: Type of analysis to perform
            
        Returns:
            Complete system prompt string
        """
        base_prompts = {
            "sentiment": "You are a sentiment analyzer. Analyze the sentiment of the text and provide insights about emotional tone, positivity/negativity, and key emotional indicators. Return your analysis in JSON format with keys: sentiment, confidence, emotional_tone, key_indicators.",
            "key_points": "You are a key points extractor. Extract and summarize the most important points from the text. Return your analysis in JSON format with keys: main_points, supporting_details, importance_ranking.",
            "summary": "You are a text summarizer. Create a concise summary of the main ideas. Return your analysis in JSON format with keys: summary, main_themes, word_count.",
            "general": "You are a comprehensive text analyzer. Provide general insights about the text including main themes, tone, structure, and key insights. Return your analysis in JSON format with keys: main_themes, tone, structure, key_insights, summary."
        }
        
        base_prompt = base_prompts.get(analysis_type, base_prompts["general"])
        
        if self.available_tools:
            tool_context = self._build_tool_context_string()
            return f"{base_prompt}\n\n{tool_context}"
        
        return base_prompt
    
    def _build_tool_context_string(self) -> str:
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
You have access to the following tools. Use them when appropriate to enhance your analysis:

{''.join(tool_descriptions)}

To use a tool, respond with a JSON object containing:
{{
    "tool_call": {{
        "tool_name": "tool_name",
        "arguments": {{"param1": "value1", "param2": "value2"}}
    }},
    "analysis": "Your analysis of the results"
}}

If you don't need tools, respond normally with your analysis.
"""
    
    def _process_ai_response(self, response: str, analysis_type: str) -> Dict[str, Any]:
        """
        Process AI response and handle tool calls if present.
        
        Args:
            response: Raw AI response string
            analysis_type: Type of analysis being performed
            
        Returns:
            Processed response dictionary
        """
        # Validate tool context if present
        if self.tool_context and not self._validate_tool_context():
            return {
                "error": "Invalid tool context structure",
                "analysis_type": analysis_type,
                "status": "error"
            }
        
        # Try to detect tool calls in the response
        tool_calls = self._extract_tool_calls(response)
        
        if tool_calls:
            # Validate each tool call
            valid_tool_calls = []
            for tool_call in tool_calls:
                if self._validate_tool_call(tool_call):
                    valid_tool_calls.append(tool_call)
                else:
                    return {
                        "error": f"Invalid tool call: {tool_call}",
                        "analysis_type": analysis_type,
                        "status": "error"
                    }
            
            if valid_tool_calls:
                return {
                    "tool_calls": valid_tool_calls,
                    "analysis_type": analysis_type,
                    "status": "tool_requested",
                    "message": "Tool execution required"
                }
        
        # No tool calls, return normal analysis
        try:
            analysis_result = json.loads(response)
            return analysis_result
        except json.JSONDecodeError:
            return {
                "analysis_type": analysis_type,
                "result": response,
                "status": "success"
            }
    
    def _extract_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from AI response.
        
        Args:
            response: AI response string to parse
            
        Returns:
            List of tool call dictionaries
        """
        tool_calls = []
        
        # First, try to parse the entire response as JSON
        try:
            full_response = json.loads(response)
            if "tool_call" in full_response:
                tool_calls.append(full_response["tool_call"])
                return tool_calls
        except json.JSONDecodeError:
            pass
        
        # Look for JSON objects containing tool_call in the response
        # Use a more flexible approach to find JSON objects
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if '"tool_call"' in line:
                try:
                    # Try to parse the line as JSON
                    obj = json.loads(line)
                    if "tool_call" in obj:
                        tool_calls.append(obj["tool_call"])
                except json.JSONDecodeError:
                    # If the line isn't complete JSON, try to find JSON within it
                    json_pattern = r'\{[^{}]*"tool_call"[^{}]*\}'
                    matches = re.findall(json_pattern, line)
                    for match in matches:
                        try:
                            tool_call_obj = json.loads(match)
                            if "tool_call" in tool_call_obj:
                                tool_calls.append(tool_call_obj["tool_call"])
                        except json.JSONDecodeError:
                            continue
        
        return tool_calls
    
    def _validate_tool_context(self) -> bool:
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
    
    def _validate_tool_call(self, tool_call: Dict[str, Any]) -> bool:
        """
        Validate tool call structure.
        
        Args:
            tool_call: Tool call dictionary to validate
            
        Returns:
            True if tool call is valid, False otherwise
        """
        if not isinstance(tool_call, dict):
            return False
        
        if "tool_name" not in tool_call:
            return False
        
        if not isinstance(tool_call["tool_name"], str):
            return False
        
        if tool_call["tool_name"] not in self.available_tools:
            return False
        
        if "arguments" in tool_call:
            if not isinstance(tool_call["arguments"], dict):
                return False
        
        return True
    
    def _execute_tools_and_analyze(self, tool_calls: List[Dict[str, Any]], text: str, analysis_type: str, client, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Execute tool calls and generate final analysis with tool results.
        
        Args:
            tool_calls: List of tool calls to execute
            text: Original text being analyzed
            analysis_type: Type of analysis being performed
            client: AI client for generating final analysis
            messages: Original conversation messages
            
        Returns:
            Final analysis results with tool information integrated
        """
        try:
            # Execute each tool call and collect results
            tool_results = []
            for tool_call in tool_calls:
                tool_result = self._execute_single_tool(tool_call)
                tool_results.append({
                    "tool_name": tool_call["tool_name"],
                    "arguments": tool_call.get("arguments", {}),
                    "result": tool_result,
                    "success": tool_result.get("success", False)
                })
            
            # Build context with tool results
            tool_context_info = self._build_tool_results_context(tool_results)
            
            # Create enhanced system prompt for final analysis
            final_system_prompt = f"""You are a comprehensive text analyzer. Provide analysis including main themes, tone, structure, and key insights. Return your analysis in JSON format with keys: main_themes, tone, structure, key_insights, summary.

IMPORTANT: You have access to the following tool results that should be integrated into your analysis:

{tool_context_info}

Use this information to enhance your analysis and provide more accurate, up-to-date insights. Mention which tools provided the information when relevant."""

            # Generate final analysis with tool results
            final_messages = [
                {"role": "system", "content": final_system_prompt},
                {"role": "user", "content": f"Analyze this text with the provided tool information:\n{text}"}
            ]
            
            final_response = client.chat.completions.create(
                model="openai:gpt-4o",
                messages=final_messages,
                temperature=0.1
            )
            
            # Parse and return final analysis
            try:
                final_analysis = json.loads(final_response.choices[0].message.content)
                # Add tool information to the result
                final_analysis["tool_results"] = tool_results
                final_analysis["tools_used"] = [tr["tool_name"] for tr in tool_results if tr["success"]]
                return final_analysis
            except json.JSONDecodeError:
                return {
                    "analysis_type": analysis_type,
                    "result": final_response.choices[0].message.content,
                    "tool_results": tool_results,
                    "tools_used": [tr["tool_name"] for tr in tool_results if tr["success"]],
                    "status": "success"
                }
                
        except Exception as e:
            return {
                "error": f"Error executing tools and analyzing: {str(e)}",
                "analysis_type": analysis_type,
                "status": "error"
            }
    
    def _execute_single_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single tool call.
        
        Args:
            tool_call: Tool call dictionary with tool_name and arguments
            
        Returns:
            Tool execution result
        """
        tool_name = tool_call["tool_name"]
        arguments = tool_call.get("arguments", {})
        
        # Mock tool execution - in real implementation, this would call actual tools
        # For now, we'll simulate tool responses based on tool type
        if tool_name == "web_search":
            return {
                "success": True,
                "query": arguments.get("query", ""),
                "results": [
                    {
                        "title": f"Search results for: {arguments.get('query', '')}",
                        "snippet": f"Latest information about {arguments.get('query', '')}",
                        "url": "https://example.com"
                    }
                ]
            }
        elif tool_name == "data_analyzer":
            return {
                "success": True,
                "data": arguments.get("data", ""),
                "analysis_type": arguments.get("analysis_type", "general"),
                "insights": [
                    "Data shows interesting patterns",
                    "Statistical analysis reveals trends",
                    "Key metrics identified"
                ]
            }
        else:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}"
            }
    
    def _build_tool_results_context(self, tool_results: List[Dict[str, Any]]) -> str:
        """
        Build context string from tool results.
        
        Args:
            tool_results: List of tool execution results
            
        Returns:
            Formatted context string for AI
        """
        context_parts = []
        
        for i, result in enumerate(tool_results, 1):
            tool_name = result["tool_name"]
            success = result["success"]
            
            if success:
                context_parts.append(f"""
Tool {i}: {tool_name}
Status: Success
Arguments: {result['arguments']}
Result: {json.dumps(result['result'], indent=2)}
""")
            else:
                context_parts.append(f"""
Tool {i}: {tool_name}
Status: Failed
Error: {result['result'].get('error', 'Unknown error')}
""")
        
        return "\n".join(context_parts)
    
    def analyze_text(self, text: str, analysis_type: str = "general") -> Dict[str, Any]:
        """
        Analyze text and provide insights using AI with tool integration.
        
        Args:
            text: Text content to analyze
            analysis_type: Type of analysis (sentiment, key_points, summary)
            
        Returns:
            Analysis results with insights as a dictionary
        """
        try:
            import aisuite as ai
            from dotenv import load_dotenv
            load_dotenv()
            
            client = ai.Client()
            
            # Use the enhanced system prompt generation
            system_prompt = self._build_system_prompt(analysis_type)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this text:\n{text}"}
            ]
            
            response = client.chat.completions.create(
                model="openai:gpt-4o",
                messages=messages,
                temperature=0.1
            )
            

            # Process AI response and handle tool calls if present
            result = self._process_ai_response(response.choices[0].message.content, analysis_type)



            # If tool calls are requested, execute them and generate final analysis
            if result.get("status") == "tool_requested" and "tool_calls" in result:
                return self._execute_tools_and_analyze(result["tool_calls"], text, analysis_type, client, messages)
            
            return result
            
        except Exception as e:
            return {
                "error": f"Error analyzing text: {str(e)}",
                "analysis_type": analysis_type,
                "status": "error"
            }
    
    def summarize_content(self, content: str, max_length: int = 200) -> str:
        """
        Create a summary of content using AI.
        
        Args:
            content: Content to summarize
            max_length: Maximum summary length
            
        Returns:
            Summarized content as a string
        """
        try:
            import aisuite as ai
            from dotenv import load_dotenv
            load_dotenv()
            
            client = ai.Client()
            messages = [
                {"role": "system", "content": f"You are a content summarizer. Create a concise summary of the content in approximately {max_length} characters or less. Focus on the main ideas and key information."},
                {"role": "user", "content": f"Summarize this content:\n{content}"}
            ]
            
            response = client.chat.completions.create(
                model="openai:gpt-4o",
                messages=messages,
                temperature=0.1
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error summarizing content: {str(e)}. Please check your API key and internet connection."

def main():
    """Main entry point for agent execution."""
    if len(sys.argv) != 2:
        print(json.dumps({"error": "Invalid arguments"}))
        sys.exit(1)
    
    try:
        # Parse input from command line
        input_data = json.loads(sys.argv[1])
        method = input_data.get("method")
        parameters = input_data.get("parameters", {})
        tool_context = input_data.get("tool_context", {})
        
        # Create agent instance with tool context
        agent = AnalysisAgent(tool_context=tool_context)
        
        # Execute requested method
        if method == "analyze_text":
            result = agent.analyze_text(
                parameters.get("text", ""),
                parameters.get("analysis_type", "general")
            )
            print(json.dumps({"result": result}))
        elif method == "summarize_content":
            result = agent.summarize_content(
                parameters.get("content", ""),
                parameters.get("max_length", 200)
            )
            print(json.dumps({"result": result}))
        else:
            print(json.dumps({"error": f"Unknown method: {method}"}))
            sys.exit(1)
            
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
