#!/usr/bin/env python3
"""
Modular Analysis Agent
AI-driven analysis agent with problem decomposition and tool execution capabilities.
"""

import json
import sys
from typing import Dict, Any, List, Optional

# Custom print function to avoid JSON parsing issues
def log_print(*args, **kwargs):
    """Print to stderr to avoid interfering with JSON output"""
    print(*args, file=sys.stderr, **kwargs)

# Import modular components
from agent_modules.core.base_agent import BaseAgent
from agent_modules.planning.problem_decomposer import ProblemDecomposer
from agent_modules.planning.step_planner import StepPlanner
from agent_modules.execution.tool_executor import ToolExecutor
from agent_modules.analysis.text_analyzer import TextAnalyzer
from agent_modules.analysis.result_aggregator import ResultAggregator
from agent_modules.utils.response_parser import ResponseParser
from agent_modules.utils.ai_client import AIClientWrapper


class ModularAnalysisAgent(BaseAgent):
    """Modular analysis agent with AI-driven planning and execution."""
    
    def __init__(self, tool_context: Optional[Dict[str, Any]] = None):
        """
        Initialize the modular analysis agent.
        
        Args:
            tool_context: Dictionary containing tool metadata and context information
        """
        super().__init__(tool_context)
        
        # Initialize all modules
        self.problem_decomposer = ProblemDecomposer(self.available_tools, self.tool_descriptions)
        self.step_planner = StepPlanner(self.available_tools, self.tool_descriptions)
        self.tool_executor = ToolExecutor(self.available_tools, self.tool_descriptions)
        self.text_analyzer = TextAnalyzer()
        self.result_aggregator = ResultAggregator()
        self.response_parser = ResponseParser()
        self.ai_client = AIClientWrapper()
        
        log_print(f"🤖 Modular Analysis Agent initialized with {len(self.available_tools)} tools")
    
    def solve_problem(self, problem: str, context: str = "") -> Dict[str, Any]:
        """
        Break down a complex problem into steps and execute each step with tool selection.
        
        Args:
            problem: The problem statement to solve
            context: Additional context or constraints
            
        Returns:
            Complete solution with step-by-step execution results
        """
        print("🧩 AGENT PROBLEM SOLVING MODE", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"📝 Problem: {problem}", file=sys.stderr)
        print(f"📋 Context: {context if context else 'None provided'}", file=sys.stderr)
        print(f"🔧 Available tools: {self.available_tools}", file=sys.stderr)
        
        try:
            # Step 1: Break down the problem into steps
            print("\n🔍 STEP 1: Problem Decomposition", file=sys.stderr)
            steps = self.problem_decomposer.decompose_problem(problem, context)
            
            if not steps or "error" in steps:
                return steps
            
            print(f"   ✅ Problem broken into {len(steps['steps'])} steps", file=sys.stderr)
            for i, step in enumerate(steps['steps'], 1):
                print(f"   {i}. {step['description']}", file=sys.stderr)
            
            # Step 2: Execute each step
            print(f"\n🚀 STEP 2: Sequential Step Execution", file=sys.stderr)
            execution_results = self._execute_problem_steps(steps['steps'], problem, context)
            
            # Step 3: Aggregate results
            print(f"\n📊 STEP 3: Result Aggregation", file=sys.stderr)
            final_solution = self.result_aggregator.aggregate_step_results(execution_results, problem, steps['steps'])
            
            return {
                "problem": problem,
                "context": context,
                "decomposition": steps,
                "step_executions": execution_results,
                "final_solution": final_solution,
                "status": "completed",
                "tools_used": list(set([tool for result in execution_results if result.get('tools_used') for tool in result['tools_used']]))
            }
            
        except Exception as e:
            log_print(f"\n❌ PROBLEM SOLVING ERROR: {str(e)}")
            return {
                "error": f"Error solving problem: {str(e)}",
                "problem": problem,
                "status": "error"
            }
    
    def analyze_text(self, text: str, analysis_type: str = "general") -> Dict[str, Any]:
        """
        Analyze text and provide insights using AI with tool integration.
        
        Args:
            text: Text content to analyze
            analysis_type: Type of analysis (sentiment, key_points, summary, mathematical)
            
        Returns:
            Analysis results with insights as a dictionary
        """
        log_print("🤖 AGENT STARTING ANALYSIS")
        log_print("=" * 50)
        log_print(f"📝 Text to analyze: {text[:100]}{'...' if len(text) > 100 else ''}")
        log_print(f"🎯 Analysis type: {analysis_type}")
        log_print(f"🔧 Available tools: {self.available_tools}")
        log_print(f"📋 Tool context loaded: {bool(self.tool_context)}")
        
        try:
            # Use the enhanced system prompt generation
            log_print("\n🧠 AGENT AI PROCESSING: Building system prompt with tool context...")
            system_prompt = self._build_system_prompt(analysis_type)
            log_print(f"   📝 System prompt built with {len(self.available_tools)} tools")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this text:\n{text}"}
            ]
            
            log_print("   🚀 Sending request to AI model...")
            response_text = self.ai_client.generate_response_with_messages(messages)
            log_print(f"   ✅ AI response received: {len(response_text)} characters")

            # Process AI response and handle tool calls if present
            log_print("\n" + "="*50)
            result = self._process_ai_response(response_text, analysis_type)
            log_print("="*50)

            # If tool calls are requested, execute them and generate final analysis
            if result.get("status") == "tool_requested" and "tool_calls" in result:
                log_print(f"\n🚀 AGENT WORKFLOW: Tool execution required ({len(result['tool_calls'])} tools)")
                return self.tool_executor.execute_tools_workflow(
                    result["tool_calls"], text, analysis_type, self.ai_client._get_client(), messages
                )
            else:
                log_print("\n📄 AGENT WORKFLOW: No tools needed, returning direct analysis")
            
            return result
            
        except Exception as e:
            log_print(f"\n❌ AGENT ERROR: {str(e)}")
            return {
                "error": f"Error analyzing text: {str(e)}",
                "analysis_type": analysis_type,
                "tool_results": [],
                "tools_used": [],
                "total_steps": 0,
                "accumulated_context": "",
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
        return self.text_analyzer.summarize_content(content, max_length)
    
    def _build_system_prompt(self, analysis_type: str) -> str:
        """
        Build system prompt with tool context if available.
        
        Args:
            analysis_type: Type of analysis to perform
            
        Returns:
            Complete system prompt string
        """
        # Use text analyzer to build base prompt
        base_prompt = self.text_analyzer._build_analysis_prompt(analysis_type)
        
        if self.available_tools:
            tool_context = self.build_tool_context_string()
            return f"{base_prompt}\n\n{tool_context}"
        
        return base_prompt
    
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
        if self.tool_context and not self.validate_tool_context():
            return {
                "error": "Invalid tool context structure",
                "analysis_type": analysis_type,
                "tool_results": [],
                "tools_used": [],
                "total_steps": 0,
                "accumulated_context": "",
                "status": "error"
            }
        
        # Try to detect tool calls in the response
        tool_calls = self.response_parser.extract_tool_calls(response)
        
        if tool_calls:
            # Validate each tool call
            from agent_modules.utils.tool_validator import ToolValidator
            validator = ToolValidator(self.available_tools)
            validation_result = validator.validate_tool_calls_batch(tool_calls)
            
            if validation_result["all_valid"]:
                return {
                    "tool_calls": validation_result["valid_calls"],
                    "analysis_type": analysis_type,
                    "status": "tool_requested",
                    "message": "Tool execution required"
                }
            else:
                return {
                    "error": f"Invalid tool calls found: {validation_result['invalid_calls']}",
                    "analysis_type": analysis_type,
                    "tool_results": [],
                    "tools_used": [],
                    "total_steps": 0,
                    "accumulated_context": "",
                    "status": "error"
                }
        
        # No tool calls, return normal analysis with consistent format
        try:
            # Clean up markdown code blocks if present
            response_content = response
            if "```json" in response_content:
                response_content = response_content.replace("```json", "").replace("```", "").strip()
            elif "```" in response_content:
                response_content = response_content.replace("```", "").strip()
            
            analysis_result = json.loads(response_content)
            # Ensure consistent output format even without tools
            if isinstance(analysis_result, dict):
                # Add missing fields to match tool output format
                analysis_result.setdefault("tool_results", [])
                analysis_result.setdefault("tools_used", [])
                analysis_result.setdefault("total_steps", 0)
                analysis_result.setdefault("accumulated_context", "")
                analysis_result.setdefault("status", "success")
            return analysis_result
        except json.JSONDecodeError:
            # Return consistent format even for non-JSON responses
            return {
                "analysis_type": analysis_type,
                "result": response,
                "tool_results": [],
                "tools_used": [],
                "total_steps": 0,
                "accumulated_context": "",
                "status": "success"
            }
    
    def _execute_problem_steps(self, steps: List[Dict[str, Any]], original_problem: str, context: str) -> List[Dict[str, Any]]:
        """
        Execute each step of the problem solution sequentially.
        
        Args:
            steps: List of step dictionaries from decomposition
            original_problem: The original problem statement
            context: Additional context
            
        Returns:
            List of execution results for each step
        """
        execution_results = []
        accumulated_results = {}  # Store results from previous steps
        
        for i, step in enumerate(steps):
            step_num = step.get('step_number', i + 1)
            log_print(f"\n   🔄 EXECUTING STEP {step_num}: {step['description']}")
            
            # Build context with previous results
            step_context = self.step_planner.build_step_context(step, accumulated_results, original_problem, context)
            
            # Execute the step
            step_result = self._execute_single_step(step, step_context)
            
            # Store result for future steps
            accumulated_results[step_num] = step_result
            execution_results.append({
                "step_number": step_num,
                "step_description": step['description'],
                "execution_result": step_result,
                "status": step_result.get("status", "completed"),
                "tools_used": step_result.get("tools_used", [])
            })
            
            log_print(f"   ✅ Step {step_num} completed: {step_result.get('status', 'success')}")
            
            # If step failed, decide whether to continue or stop
            if step_result.get("status") == "error":
                log_print(f"   ⚠️ Step {step_num} failed, but continuing with remaining steps...")
        
        return execution_results
    
    def _execute_single_step(self, step: Dict[str, Any], step_context: str) -> Dict[str, Any]:
        """
        Execute a single step of the problem solution.
        
        Args:
            step: Step dictionary with execution details
            step_context: Context for this step execution
            
        Returns:
            Step execution result
        """
        try:
            # Build step-specific system prompt
            system_prompt = self.step_planner.build_step_execution_prompt(step)
            
            log_print(f"      🧠 AI executing step: {step.get('type', 'general')}")
            response_text = self.ai_client.generate_response(system_prompt, step_context)
            
            # Process AI response and handle tool calls if present
            result = self._process_ai_response(response_text, f"step_{step.get('step_number', 'unknown')}")
            
            # If tool calls are requested, execute them
            if result.get("status") == "tool_requested" and "tool_calls" in result:
                log_print(f"      🔧 Step requires {len(result['tool_calls'])} tools")
                tool_result = self.tool_executor.execute_tools_workflow(
                    result["tool_calls"], step_context, "step_execution", 
                    self.ai_client._get_client(), []
                )
                return {
                    "status": "completed",
                    "result": tool_result,
                    "tools_used": tool_result.get("tools_used", []),
                    "summary": self.response_parser.extract_step_summary(tool_result)
                }
            else:
                return {
                    "status": "completed", 
                    "result": result,
                    "tools_used": [],
                    "summary": self.response_parser.extract_step_summary(result)
                }
                
        except Exception as e:
            return {
                "status": "error",
                "error": f"Error executing step: {str(e)}",
                "summary": f"Step failed: {str(e)}"
            }


def main():
    """Main entry point for modular agent execution."""
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
        agent = ModularAnalysisAgent(tool_context=tool_context)
        
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
        elif method == "solve_problem":
            result = agent.solve_problem(
                parameters.get("problem", ""),
                parameters.get("context", "")
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
