#!/usr/bin/env python3
"""
Problem Decomposer Module
Handles AI-driven problem decomposition and step planning.
"""

import json
import re
from typing import Dict, Any, List
from ..utils.ai_client import AIClientWrapper


class ProblemDecomposer:
    """Handles decomposition of complex problems into manageable steps."""
    
    def __init__(self, available_tools: List[str] = None, tool_descriptions: Dict[str, str] = None):
        """
        Initialize the problem decomposer.
        
        Args:
            available_tools: List of available tool names
            tool_descriptions: Dictionary mapping tool names to descriptions
        """
        self.available_tools = available_tools or []
        self.tool_descriptions = tool_descriptions or {}
        self.ai_client = AIClientWrapper()
    
    def decompose_problem(self, problem: str, context: str = "") -> Dict[str, Any]:
        """
        Use AI to break down a complex problem into manageable steps.
        
        Args:
            problem: The problem to decompose
            context: Additional context
            
        Returns:
            Dictionary with decomposed steps
        """
        try:
            # Build tool-aware decomposition prompt
            tool_context = self._build_tool_context() if self.available_tools else ""
            
            system_prompt = f"""You are an expert problem decomposer. Break down complex problems into clear, sequential steps that can be executed one by one.

{tool_context}

CRITICAL INSTRUCTIONS:
1. Analyze the problem and break it into 3-7 logical steps
2. Each step should be actionable and specific
3. If a step requires mathematical calculations and you have math tools, note that tools will be used
4. If a step requires text processing and you have text tools, note that tools will be used
5. Steps should build upon each other logically
6. Include any data gathering, processing, calculation, and analysis steps needed

Return your response as a JSON object with this exact structure:
{{
    "analysis": "Brief analysis of the problem",
    "complexity": "low|medium|high",
    "estimated_steps": "number",
    "steps": [
        {{
            "step_number": 1,
            "description": "Clear description of what this step does",
            "type": "data_gathering|calculation|analysis|processing|synthesis",
            "requires_tools": true/false,
            "expected_tools": ["tool1", "tool2"] or [],
            "depends_on": [] or [step_numbers],
            "expected_output": "What this step should produce"
        }}
    ],
    "success_criteria": "How to know if the problem is solved"
}}

EXAMPLES:
- Math problems → steps for data extraction, calculation, verification
- Text analysis → steps for reading, processing, analyzing, summarizing
- Multi-part questions → steps for each part plus synthesis"""

            print("   🧠 AI analyzing problem structure...")
            response_text = self.ai_client.generate_response(
                system_prompt,
                f"Problem: {problem}\nContext: {context}\n\nPlease decompose this problem into clear steps."
            )
            
            print(f"   ✅ Problem decomposition received")
            
            # Parse the decomposition
            try:
                decomposition = json.loads(response_text)
                return decomposition
            except json.JSONDecodeError:
                print("   ⚠️ Failed to parse JSON, trying to extract...")
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    decomposition = json.loads(json_match.group())
                    return decomposition
                else:
                    return {"error": "Could not parse problem decomposition", "raw_response": response_text}
                    
        except Exception as e:
            return {"error": f"Error decomposing problem: {str(e)}"}
    
    def _build_tool_context(self) -> str:
        """Build tool context for decomposition prompt."""
        if not self.available_tools:
            return ""
        
        tool_info = []
        for tool_name in self.available_tools:
            description = self.tool_descriptions.get(tool_name, f"Tool: {tool_name}")
            tool_info.append(f"- {tool_name}: {description}")
        
        return f"""
AVAILABLE TOOLS:
{chr(10).join(tool_info)}

When decomposing problems, consider which steps will require these tools.
"""
    
    def validate_decomposition(self, decomposition: Dict[str, Any]) -> bool:
        """
        Validate that a decomposition has the expected structure.
        
        Args:
            decomposition: Decomposition dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(decomposition, dict):
            return False
        
        required_keys = ["analysis", "complexity", "steps"]
        if not all(key in decomposition for key in required_keys):
            return False
        
        if not isinstance(decomposition["steps"], list):
            return False
        
        # Validate each step
        for step in decomposition["steps"]:
            if not isinstance(step, dict):
                return False
            
            step_required_keys = ["step_number", "description", "type"]
            if not all(key in step for key in step_required_keys):
                return False
        
        return True
    
    def get_step_dependencies(self, decomposition: Dict[str, Any]) -> Dict[int, List[int]]:
        """
        Extract step dependencies from decomposition.
        
        Args:
            decomposition: Problem decomposition
            
        Returns:
            Dictionary mapping step numbers to their dependencies
        """
        dependencies = {}
        
        if "steps" in decomposition:
            for step in decomposition["steps"]:
                step_num = step.get("step_number", 0)
                deps = step.get("depends_on", [])
                dependencies[step_num] = deps
        
        return dependencies
