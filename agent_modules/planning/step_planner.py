#!/usr/bin/env python3
"""
Step Planner Module
Handles dynamic step-by-step planning and continuation logic.
"""

import sys
from typing import Dict, Any, List
from ..utils.ai_client import AIClientWrapper, get_shared_ai_client


class StepPlanner:
    """Handles dynamic planning of next steps based on current progress."""
    
    def __init__(self, available_tools: List[str] = None, tool_descriptions: Dict[str, str] = None):
        """
        Initialize the step planner.
        
        Args:
            available_tools: List of available tool names
            tool_descriptions: Dictionary mapping tool names to descriptions
        """
        self.available_tools = available_tools or []
        self.tool_descriptions = tool_descriptions or {}
        self.ai_client = get_shared_ai_client()
    
    def check_for_continuation(self, accumulated_context: str, original_text: str, previous_failures: int = 0) -> str:
        """
        Check if the AI wants to continue with more tool executions based on current progress.
        
        Args:
            accumulated_context: Context from previous steps
            original_text: Original request text
            previous_failures: Number of consecutive failures
            
        Returns:
            AI response indicating next steps
        """
        tool_context = self._build_tool_context() if self.available_tools else ""
        
        # Check for repeated failures - stop if tools are failing
        if previous_failures >= 2:
            return "COMPLETE - Tools are failing, stopping execution"
        
        continuation_prompt = f"""You are continuing a multi-step calculation/analysis. 

{tool_context}

CURRENT PROGRESS:
{accumulated_context}

ORIGINAL REQUEST: {original_text}

INSTRUCTIONS:
1. Look at the original request and current progress
2. Determine if more tool executions are needed to complete the request
3. If tools are failing repeatedly, respond with "COMPLETE" 
4. If yes, provide the next tool call(s) needed
5. If no, respond with "COMPLETE" 

If more tools are needed, respond with a JSON object containing the tool_call:
{{
    "tool_call": {{
        "tool_name": "tool_name",
        "arguments": {{"param1": "value1", "param2": "value2"}}
    }},
    "reasoning": "Why this tool is needed next"
}}

If the task is complete or tools are failing, just respond with: "COMPLETE"
"""

        try:
            response_text = self.ai_client.generate_response(
                continuation_prompt,
                "What should be the next step?"
            )
            
            print(f"   🧠 Continuation check response: {response_text[:100]}...", file=sys.stderr)
            return response_text
            
        except Exception as e:
            print(f"   ⚠️ Error checking continuation: {e}", file=sys.stderr)
            return "COMPLETE"
    
    def build_step_context(self, step: Dict[str, Any], previous_results: Dict[int, Any], 
                          problem: str, context: str) -> str:
        """
        Build context for a step execution including previous results.
        
        Args:
            step: Current step dictionary
            previous_results: Results from previous steps
            problem: Original problem
            context: Additional context
            
        Returns:
            Formatted context string
        """
        context_parts = [
            f"ORIGINAL PROBLEM: {problem}",
            f"ADDITIONAL CONTEXT: {context}" if context else "",
            f"CURRENT STEP: {step['description']}",
            f"STEP TYPE: {step.get('type', 'general')}",
        ]
        
        # Add dependency results
        depends_on = step.get('depends_on', [])
        if depends_on and previous_results:
            context_parts.append("\nPREVIOUS STEP RESULTS:")
            for dep_step in depends_on:
                if dep_step in previous_results:
                    result = previous_results[dep_step]
                    summary = self._extract_result_summary(result)
                    context_parts.append(f"  Step {dep_step}: {summary}")
        
        # Add all previous results if no specific dependencies
        elif previous_results and not depends_on:
            context_parts.append("\nALL PREVIOUS RESULTS:")
            for step_num, result in previous_results.items():
                summary = self._extract_result_summary(result)
                context_parts.append(f"  Step {step_num}: {summary}")
        
        return "\n".join(filter(None, context_parts))
    
    def build_step_execution_prompt(self, step: Dict[str, Any]) -> str:
        """
        Build system prompt for executing a specific step.
        
        Args:
            step: Step dictionary
            
        Returns:
            System prompt for step execution
        """
        base_prompt = f"""You are executing step {step.get('step_number', 'unknown')} of a multi-step problem solution.

STEP DETAILS:
- Description: {step['description']}
- Type: {step.get('type', 'general')}
- Expected Output: {step.get('expected_output', 'Analysis result')}

"""

        if step.get('requires_tools', False) and self.available_tools:
            tool_context = self._build_tool_context()
            return base_prompt + tool_context
        else:
            return base_prompt + """
Provide a clear, detailed response for this step. If this involves calculations, show your work.
If this involves analysis, provide specific insights. Format your response clearly and completely.

Return your response in JSON format with keys: analysis, result, insights, next_steps (if applicable).
"""
    
    def _build_tool_context(self) -> str:
        """Build tool context for prompts."""
        if not self.available_tools:
            return ""
        
        tool_info = []
        for tool_name in self.available_tools:
            description = self.tool_descriptions.get(tool_name, f"Tool: {tool_name}")
            tool_info.append(f"- {tool_name}: {description}")
        
        return f"""
AVAILABLE TOOLS:
{chr(10).join(tool_info)}

To use a tool, respond with a JSON object containing the tool_call:
{{
    "tool_call": {{
        "tool_name": "tool_name",
        "arguments": {{"a": "value1", "b": "value2"}}
    }},
    "reasoning": "Why this tool is needed"
}}

IMPORTANT: Use parameter names like "a", "b" for mathematical operations, or specific parameter names as appropriate for each tool type.
"""
    
    def _extract_result_summary(self, result: Dict[str, Any]) -> str:
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
