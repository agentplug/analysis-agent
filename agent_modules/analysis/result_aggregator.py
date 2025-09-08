#!/usr/bin/env python3
"""
Result Aggregator Module
Handles aggregation of results from multiple steps into final solutions.
"""

import json
from typing import Dict, Any, List
from ..utils.ai_client import AIClientWrapper


class ResultAggregator:
    """Aggregates results from multiple execution steps into final solutions."""
    
    def __init__(self):
        """Initialize result aggregator."""
        self.ai_client = AIClientWrapper()
    
    def aggregate_step_results(self, execution_results: List[Dict[str, Any]], 
                             problem: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from all steps into a final solution.
        
        Args:
            execution_results: Results from step executions
            problem: Original problem
            steps: Original step definitions
            
        Returns:
            Final aggregated solution
        """
        try:
            # Build aggregation context
            results_summary = self._build_results_summary(execution_results)
            
            system_prompt = """You are a solution synthesizer. Take the results from multiple problem-solving steps and create a comprehensive final solution.

INSTRUCTIONS:
1. Analyze all step results
2. Identify the key findings and outputs
3. Synthesize a complete answer to the original problem
4. Highlight any tool-generated results
5. Note any limitations or assumptions

Return your response as JSON with keys: final_answer, key_findings, methodology_summary, confidence_level, limitations."""

            user_message = f"ORIGINAL PROBLEM: {problem}\n\nSTEP EXECUTION RESULTS:\n{results_summary}\n\nPlease synthesize these results into a final comprehensive solution."
            
            print("   🧠 AI synthesizing final solution...")
            response_text = self.ai_client.generate_response(system_prompt, user_message)
            
            try:
                final_solution = json.loads(response_text)
                return final_solution
            except json.JSONDecodeError:
                return {
                    "final_answer": response_text,
                    "methodology_summary": f"Problem solved in {len(execution_results)} steps",
                    "confidence_level": "high",
                    "raw_synthesis": response_text
                }
                
        except Exception as e:
            return {
                "error": f"Error aggregating results: {str(e)}",
                "step_count": len(execution_results),
                "completed_steps": len([r for r in execution_results if r.get('status') != 'error'])
            }
    
    def _build_results_summary(self, execution_results: List[Dict[str, Any]]) -> str:
        """
        Build summary of execution results.
        
        Args:
            execution_results: List of execution results
            
        Returns:
            Formatted summary string
        """
        summary_parts = []
        
        for result in execution_results:
            step_num = result.get('step_number', 'unknown')
            description = result.get('step_description', 'Unknown step')
            status = result.get('status', 'unknown')
            tools_used = result.get('tools_used', [])
            
            # Extract result summary
            execution_result = result.get('execution_result', {})
            result_summary = execution_result.get('summary', 'Completed')
            
            summary_parts.append(f"""Step {step_num}: {description}
  Result: {result_summary}
  Status: {status}
  Tools Used: {', '.join(tools_used) if tools_used else 'None'}""")
        
        return "\n".join(summary_parts)
    
    def build_tool_results_context(self, tool_results: List[Dict[str, Any]]) -> str:
        """
        Build context string from tool results.
        
        Args:
            tool_results: List of tool execution results
            
        Returns:
            Formatted context string for AI
        """
        context_parts = []
        
        for i, result in enumerate(tool_results, 1):
            tool_name = result.get("tool_name", "unknown")
            success = result.get("success", False)
            
            if success:
                context_parts.append(f"""
Tool {i}: {tool_name}
Status: Success
Arguments: {result.get('arguments', {})}
Result: {json.dumps(result.get('result', {}), indent=2)}
""")
            else:
                error_info = result.get('result', {})
                error_msg = error_info.get('error', 'Unknown error') if isinstance(error_info, dict) else str(error_info)
                context_parts.append(f"""
Tool {i}: {tool_name}
Status: Failed
Error: {error_msg}
""")
        
        return "\n".join(context_parts)
    
    def extract_final_value(self, execution_results: List[Dict[str, Any]]) -> Any:
        """
        Extract the final computed value from execution results.
        
        Args:
            execution_results: List of execution results
            
        Returns:
            Final computed value or None
        """
        # Look for the last successful result
        for result in reversed(execution_results):
            if result.get('status') != 'error':
                execution_result = result.get('execution_result', {})
                if isinstance(execution_result, dict) and 'result' in execution_result:
                    return execution_result['result']
        
        return None
    
    def calculate_success_rate(self, execution_results: List[Dict[str, Any]]) -> float:
        """
        Calculate success rate of step executions.
        
        Args:
            execution_results: List of execution results
            
        Returns:
            Success rate as float between 0 and 1
        """
        if not execution_results:
            return 0.0
        
        successful_steps = len([r for r in execution_results if r.get('status') != 'error'])
        return successful_steps / len(execution_results)
