#!/usr/bin/env python3
"""
Tool Executor Module
Handles tool execution, validation, and multi-step workflows.
"""

import asyncio
import sys
import concurrent.futures
from typing import Dict, Any, List
from ..utils.tool_validator import ToolValidator
from .mcp_client import MCPClient


class ToolExecutor:
    """Handles execution of tool calls with validation and MCP integration."""
    
    def __init__(self, available_tools: List[str] = None, tool_descriptions: Dict[str, str] = None):
        """
        Initialize the tool executor.
        
        Args:
            available_tools: List of available tool names
            tool_descriptions: Dictionary mapping tool names to descriptions
        """
        self.available_tools = available_tools or []
        self.tool_descriptions = tool_descriptions or {}
        self.validator = ToolValidator(self.available_tools)
        self.mcp_client = MCPClient(allow_fallback=False)  # Disable fallback by default
    
    def set_fallback_enabled(self, enabled: bool):
        """
        Enable or disable local fallback when MCP server is unavailable.
        
        Args:
            enabled: Whether to allow local fallback execution
        """
        self.mcp_client.allow_fallback = enabled
        print(f"🔧 MCP fallback {'enabled' if enabled else 'disabled'}", file=sys.stderr)
    
    def execute_tools_workflow(self, tool_calls: List[Dict[str, Any]], text: str, 
                             analysis_type: str, client, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Execute tool calls and continue with multi-step execution as needed.
        
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
            # Execute multi-step tool workflow
            all_tool_results = []
            current_step = 1
            max_steps = 10  # Prevent infinite loops
            consecutive_failures = 0  # Track consecutive tool failures
            
            # Keep track of accumulated results for context
            accumulated_context = f"Original request: {text}\n\n"
            
            # Start with initial tool calls
            remaining_tool_calls = tool_calls.copy()
            
            while remaining_tool_calls and current_step <= max_steps:
                print(f"\n🔄 MULTI-STEP EXECUTION: Step {current_step}", file=sys.stderr)
                
                # Execute current batch of tool calls
                step_results = []
                step_has_success = False
                
                for tool_call in remaining_tool_calls:
                    tool_result = self.execute_single_tool(tool_call)
                    print(f"\033[91m🔧 AGENT TOOL EXECUTION: Tool result: {tool_result}\033[0m", file=sys.stderr)
                    step_result = {
                        "tool_name": tool_call["tool_name"],
                        "arguments": tool_call.get("arguments", {}),
                        "result": tool_result,
                        "success": tool_result.get("success", False),
                        "step": current_step
                    }
                    step_results.append(step_result)
                    all_tool_results.append(step_result)
                    
                    # Track if any tool in this step succeeded
                    if step_result["success"]:
                        step_has_success = True
                        result_value = self._extract_tool_result_value(tool_result)
                        accumulated_context += f"Step {current_step}: {tool_call['tool_name']}({tool_call.get('arguments', {})}) = {result_value}\n"
                    else:
                        # Add failure info to context
                        error_msg = tool_result.get("error", "Unknown error")
                        accumulated_context += f"Step {current_step}: {tool_call['tool_name']} FAILED - {error_msg}\n"
                
                # Update consecutive failure counter
                if step_has_success:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    print(f"   ⚠️ Step {current_step} failed (consecutive failures: {consecutive_failures})", file=sys.stderr)
                
                # Check if we need to continue with more steps
                print(f"   🤔 Checking if more steps are needed after step {current_step}...", file=sys.stderr)
                
                # Import here to avoid circular imports
                from ..planning.step_planner import StepPlanner
                step_planner = StepPlanner(self.available_tools, self.tool_descriptions)
                continuation_response = step_planner.check_for_continuation(accumulated_context, text, consecutive_failures)
                
                # Extract any new tool calls from continuation response
                from ..utils.response_parser import ResponseParser
                parser = ResponseParser()
                remaining_tool_calls = parser.extract_tool_calls(continuation_response)
                
                if remaining_tool_calls:
                    print(f"   ✅ Found {len(remaining_tool_calls)} more tools to execute in next step", file=sys.stderr)
                    current_step += 1
                else:
                    print(f"   🏁 No more steps needed. Proceeding to final analysis.", file=sys.stderr)
                    break
            
            if current_step > max_steps:
                print(f"   ⚠️ Reached maximum steps limit ({max_steps})", file=sys.stderr)
            
            # Build context with all tool results
            tool_context_info = self._build_tool_results_context(all_tool_results)
            
            # Create enhanced system prompt for final analysis
            final_system_prompt = f"""You are a comprehensive text analyzer. Provide analysis including main themes, tone, structure, and key insights. Return your analysis in JSON format with keys: main_themes, tone, structure, key_insights, summary.

IMPORTANT: You have executed a multi-step process with the following tool results:

{tool_context_info}

ACCUMULATED RESULTS:
{accumulated_context}

Use this information to provide a complete analysis that includes the final calculated result and the methodology used."""

            # Generate final analysis with tool results
            final_messages = [
                {"role": "system", "content": final_system_prompt},
                {"role": "user", "content": f"Provide final analysis for: {text}"}
            ]
            
            final_response = client.chat.completions.create(
                model="openai:gpt-4o",
                messages=final_messages,
                temperature=0.1
            )
            
            # Parse and return final analysis
            try:
                import json
                final_analysis = json.loads(final_response.choices[0].message.content)
                # Add tool information to the result
                final_analysis["tool_results"] = all_tool_results
                final_analysis["tools_used"] = [tr["tool_name"] for tr in all_tool_results if tr["success"]]
                final_analysis["total_steps"] = current_step
                final_analysis["accumulated_context"] = accumulated_context
                return final_analysis
            except json.JSONDecodeError:
                return {
                    "analysis_type": analysis_type,
                    "result": final_response.choices[0].message.content,
                    "tool_results": all_tool_results,
                    "tools_used": [tr["tool_name"] for tr in all_tool_results if tr["success"]],
                    "total_steps": current_step,
                    "accumulated_context": accumulated_context,
                    "status": "success"
                }
                
        except Exception as e:
            return {
                "error": f"Error executing tools and analyzing: {str(e)}",
                "analysis_type": analysis_type,
                "status": "error"
            }
    
    def execute_single_tool(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single tool call using real MCP tool execution.
        ENFORCES TOOL ASSIGNMENT LIMITS with detailed logging.
        
        Args:
            tool_call: Tool call dictionary with tool_name and arguments
            
        Returns:
            Tool execution result
        """
        tool_name = tool_call["tool_name"]
        arguments = tool_call.get("arguments", {})
        
        print(f"🔧 AGENT TOOL EXECUTION: Starting execution of '{tool_name}'", file=sys.stderr)
        print(f"   📝 Arguments: {arguments}", file=sys.stderr)
        print(f"   🔐 Available tools: {self.available_tools}", file=sys.stderr)
        
        # CRITICAL SECURITY CHECK: Double-verify tool is authorized
        if not self.validator.validate_tool_call(tool_call):
            print(f"   ❌ AUTHORIZATION FAILED: Tool '{tool_name}' not in assigned tools", file=sys.stderr)
            return {
                "success": False,
                "tool_name": tool_name,
                "arguments": arguments,
                "error": f"UNAUTHORIZED: Tool '{tool_name}' is not in assigned tools list: {self.available_tools}"
            }
        
        print(f"   ✅ AUTHORIZATION PASSED: Tool '{tool_name}' is authorized", file=sys.stderr)
        
        try:
            print(f"   🚀 EXECUTING: Calling MCP server for '{tool_name}'...", file=sys.stderr)
            # Execute real tools by calling MCP server
            result = self.mcp_client.call_tool(tool_name, arguments)
            print(f"   ✅ EXECUTION SUCCESS: Got result from '{tool_name}'", file=sys.stderr)
            print(f"   📊 Result: {result}", file=sys.stderr)
            
            return {
                "success": True,
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result
            }
        except Exception as e:
            print(f"   ❌ EXECUTION FAILED: Error in '{tool_name}': {str(e)}", file=sys.stderr)
            return {
                "success": False,
                "tool_name": tool_name,
                "arguments": arguments,
                "error": f"Tool execution failed: {str(e)}"
            }
    
    def _extract_tool_result_value(self, tool_result: Dict[str, Any]) -> str:
        """Extract the actual result value from a tool execution result."""
        if isinstance(tool_result, dict):
            if 'result' in tool_result:
                result = tool_result['result']
                if isinstance(result, dict) and 'result' in result:
                    return str(result['result'])
                else:
                    return str(result)
            elif 'success' in tool_result and tool_result['success']:
                # Look for any numeric or meaningful result
                for key in ['value', 'output', 'answer', 'calculation']:
                    if key in tool_result:
                        return str(tool_result[key])
        return str(tool_result)
    
    def _build_tool_results_context(self, tool_results: List[Dict[str, Any]]) -> str:
        """
        Build context string from tool results.
        
        Args:
            tool_results: List of tool execution results
            
        Returns:
            Formatted context string for AI
        """
        import json
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
