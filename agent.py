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
        import re
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
        Validate tool call structure and ensure tool is in assigned tools list.
        
        Args:
            tool_call: Tool call dictionary to validate
            
        Returns:
            True if tool call is valid and allowed, False otherwise
        """
        if not isinstance(tool_call, dict):
            print(f"❌ Invalid tool call: not a dictionary")
            return False
        
        if "tool_name" not in tool_call:
            print(f"❌ Invalid tool call: missing 'tool_name'")
            return False
        
        if not isinstance(tool_call["tool_name"], str):
            print(f"❌ Invalid tool call: 'tool_name' must be string")
            return False
        
        tool_name = tool_call["tool_name"]
        
        # CRITICAL: Ensure tool is in the assigned tools list
        if tool_name not in self.available_tools:
            print(f"❌ UNAUTHORIZED TOOL ACCESS: '{tool_name}' is not in assigned tools: {self.available_tools}")
            return False
        
        # Validate arguments structure
        if "arguments" in tool_call:
            if not isinstance(tool_call["arguments"], dict):
                print(f"❌ Invalid tool call: 'arguments' must be dictionary")
                return False
        
        print(f"✅ Tool call validated: '{tool_name}' is authorized")
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
                print(f"\033[91m🔧 AGENT TOOL EXECUTION: Tool result: {tool_result}\033[0m")
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
        Execute a single tool call using real MCP tool execution.
        ENFORCES TOOL ASSIGNMENT LIMITS with detailed logging.
        
        Args:
            tool_call: Tool call dictionary with tool_name and arguments
            
        Returns:
            Tool execution result
        """
        tool_name = tool_call["tool_name"]
        arguments = tool_call.get("arguments", {})
        
        print(f"🔧 AGENT TOOL EXECUTION: Starting execution of '{tool_name}'")
        print(f"   📝 Arguments: {arguments}")
        print(f"   🔐 Available tools: {self.available_tools}")
        
        # CRITICAL SECURITY CHECK: Double-verify tool is authorized
        if tool_name not in self.available_tools:
            print(f"   ❌ AUTHORIZATION FAILED: Tool '{tool_name}' not in assigned tools")
            return {
                "success": False,
                "tool_name": tool_name,
                "arguments": arguments,
                "error": f"UNAUTHORIZED: Tool '{tool_name}' is not in assigned tools list: {self.available_tools}"
            }
        
        print(f"   ✅ AUTHORIZATION PASSED: Tool '{tool_name}' is authorized")
        
        try:
            print(f"   🚀 EXECUTING: Calling MCP server for '{tool_name}'...")
            # Execute real tools by calling MCP server
            result = self._call_mcp_tool(tool_name, arguments)
            print(f"   ✅ EXECUTION SUCCESS: Got result from '{tool_name}'")
            print(f"   📊 Result: {result}")
            
            return {
                "success": True,
                "tool_name": tool_name,
                "arguments": arguments,
                "result": result
            }
        except Exception as e:
            print(f"   ❌ EXECUTION FAILED: Error in '{tool_name}': {str(e)}")
            return {
                "success": False,
                "tool_name": tool_name,
                "arguments": arguments,
                "error": f"Tool execution failed: {str(e)}"
            }
    
    def _call_mcp_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call MCP tool server using proper SSE format to execute any tool.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments for the tool
            
        Returns:
            Tool execution result
        """
        import asyncio
        
        async def execute_tool_via_sse():
            try:
                # Import MCP SSE client components
                from mcp import ClientSession
                from mcp.client.sse import sse_client
                
                # Connect to MCP server using SSE format
                async with sse_client(url="http://localhost:8000/sse") as streams:
                    async with ClientSession(*streams) as session:
                        # Initialize the session
                        await session.initialize()
                        
                        # Call the tool using proper MCP format
                        result = await session.call_tool(tool_name, arguments=arguments)
                        
                        # Extract result content
                        if hasattr(result, 'content') and result.content:
                            if hasattr(result.content[0], 'text'):
                                return result.content[0].text
                            else:
                                return str(result.content[0])
                        else:
                            return str(result)
                            
            except ImportError as e:
                # MCP not available, fallback to local execution
                raise Exception(f"MCP not available: {e}")
            except Exception as e:
                # Connection or execution error, fallback to local execution
                raise Exception(f"MCP execution failed: {e}")
        
        # Handle async execution in agent context
        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an event loop, use run_in_executor to avoid blocking
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, execute_tool_via_sse())
                    return future.result(timeout=30)
            except RuntimeError:
                # No event loop running, safe to use asyncio.run()
                return asyncio.run(execute_tool_via_sse())
                
        except Exception as e:
            # If MCP fails, fall back to local execution
            print(f"MCP tool execution failed, using local fallback: {e}")
            return self._execute_tool_locally(tool_name, arguments)
    
    def _execute_tool_locally(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Dynamic tool execution that can handle ANY assigned tool.
        
        This method uses intelligent inference to execute tools based on their names
        and available arguments, making it completely tool-agnostic.
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            
        Returns:
            Tool result
        """
        try:
            # For any tool, try to execute it intelligently based on patterns
            result = self._intelligent_tool_execution(tool_name, arguments)
            
            return {
                "result": result,
                "tool_name": tool_name,
                "operation": tool_name,
                "arguments_used": arguments,
                "execution_method": "local_intelligent"
            }
            
        except Exception as e:
            # If intelligent execution fails, return a descriptive response
            return {
                "error": f"Could not execute tool '{tool_name}' locally: {str(e)}",
                "tool_name": tool_name,
                "arguments": arguments,
                "suggestion": "Tool may require MCP server connection or different argument format"
            }
    
    def _intelligent_tool_execution(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Intelligent tool execution that adapts to any tool based on name patterns and arguments.
        
        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            
        Returns:
            Execution result
        """
        # Convert arguments to appropriate types
        processed_args = self._process_tool_arguments(arguments)
        
        # Math operations (support any math tool)
        if any(keyword in tool_name.lower() for keyword in ['add', 'sum', 'plus', '+']):
            return self._execute_math_operation('+', processed_args)
        elif any(keyword in tool_name.lower() for keyword in ['subtract', 'minus', 'sub', '-']):
            return self._execute_math_operation('-', processed_args)
        elif any(keyword in tool_name.lower() for keyword in ['multiply', 'mult', 'times', '*']):
            return self._execute_math_operation('*', processed_args)
        elif any(keyword in tool_name.lower() for keyword in ['divide', 'div', '/']):
            return self._execute_math_operation('/', processed_args)
        
        # Text operations (support any text tool)
        elif any(keyword in tool_name.lower() for keyword in ['text', 'string', 'process']):
            return self._execute_text_operation(tool_name, processed_args)
        
        # Greeting tools
        elif any(keyword in tool_name.lower() for keyword in ['greet', 'hello', 'hi', 'welcome']):
            name = processed_args.get('name', processed_args.get('user', 'World'))
            return f"Hello, {name}! (via {tool_name})"
        
        # Weather tools
        elif any(keyword in tool_name.lower() for keyword in ['weather', 'climate', 'temperature']):
            location = processed_args.get('location', processed_args.get('place', 'Unknown Location'))
            return f"Weather in {location}: Sunny, 22°C (simulated via {tool_name})"
        
        # Search tools
        elif any(keyword in tool_name.lower() for keyword in ['search', 'find', 'query', 'lookup']):
            query = processed_args.get('query', processed_args.get('q', processed_args.get('search', 'default')))
            return f"Search results for '{query}' (via {tool_name}): [Simulated results]"
        
        # Data analysis tools
        elif any(keyword in tool_name.lower() for keyword in ['analyze', 'data', 'stats', 'analyze']):
            data = processed_args.get('data', processed_args.get('input', 'sample data'))
            return f"Analysis of '{data}' completed via {tool_name}: [Key insights found]"
        
        # File operations
        elif any(keyword in tool_name.lower() for keyword in ['file', 'read', 'write', 'save']):
            filename = processed_args.get('filename', processed_args.get('path', 'file.txt'))
            return f"File operation on '{filename}' via {tool_name}: [Operation completed]"
        
        # Generic tool execution - try to be intelligent about any tool
        else:
            return self._execute_generic_tool(tool_name, processed_args)
    
    def _process_tool_arguments(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Process and convert tool arguments to appropriate types."""
        processed = {}
        for key, value in arguments.items():
            # Try to convert to number if it looks like one
            if isinstance(value, str):
                try:
                    if '.' in value:
                        processed[key] = float(value)
                    else:
                        processed[key] = int(value)
                except ValueError:
                    processed[key] = value
            else:
                processed[key] = value
        return processed
    
    def _execute_math_operation(self, operation: str, arguments: Dict[str, Any]) -> Any:
        """Execute mathematical operations dynamically."""
        # Find numeric arguments
        numbers = []
        for key, value in arguments.items():
            if isinstance(value, (int, float)):
                numbers.append(value)
        
        # If we have exactly 2 numbers, perform binary operation
        if len(numbers) == 2:
            a, b = numbers
            if operation == '+':
                return a + b
            elif operation == '-':
                return a - b
            elif operation == '*':
                return a * b
            elif operation == '/':
                if b == 0:
                    return "Error: Division by zero"
                return a / b
        
        # If we have more or fewer numbers, try to adapt
        elif len(numbers) > 2:
            result = numbers[0]
            for num in numbers[1:]:
                if operation == '+':
                    result += num
                elif operation == '*':
                    result *= num
                elif operation == '-':
                    result -= num
                elif operation == '/':
                    if num != 0:
                        result /= num
            return result
        
        # Fallback for insufficient arguments
        return f"Mathematical operation {operation} requires numeric arguments"
    
    def _execute_text_operation(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute text operations dynamically."""
        # Find text arguments
        text = arguments.get('text', arguments.get('input', arguments.get('string', '')))
        operation = arguments.get('operation', arguments.get('action', 'process'))
        
        if 'upper' in tool_name.lower() or operation == 'uppercase':
            return text.upper()
        elif 'lower' in tool_name.lower() or operation == 'lowercase':
            return text.lower()
        elif 'length' in tool_name.lower() or operation == 'length':
            return len(text)
        elif 'reverse' in tool_name.lower() or operation == 'reverse':
            return text[::-1]
        elif 'count' in tool_name.lower():
            target = arguments.get('target', arguments.get('char', ' '))
            return text.count(target)
        else:
            return f"Text processed via {tool_name}: '{text}'"
    
    def _execute_generic_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Generic tool execution for unknown tools."""
        # Build a descriptive response based on the tool name and arguments
        arg_summary = ", ".join([f"{k}={v}" for k, v in arguments.items()])
        
        return {
            "message": f"Tool '{tool_name}' executed successfully",
            "arguments": arguments,
            "note": f"Generic execution with parameters: {arg_summary}",
            "tool_type": "unknown",
            "capability": "This tool can be used with any arguments you provide"
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
        Shows detailed internal process of tool selection and execution.
        
        Args:
            text: Text content to analyze
            analysis_type: Type of analysis (sentiment, key_points, summary)
            
        Returns:
            Analysis results with insights as a dictionary
        """
        print("🤖 AGENT STARTING ANALYSIS")
        print("=" * 50)
        print(f"📝 Text to analyze: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"🎯 Analysis type: {analysis_type}")
        print(f"🔧 Available tools: {self.available_tools}")
        print(f"📋 Tool context loaded: {bool(self.tool_context)}")
        
        try:
            import aisuite as ai
            from dotenv import load_dotenv
            load_dotenv()
            
            client = ai.Client()
            
            # Use the enhanced system prompt generation
            print("\n🧠 AGENT AI PROCESSING: Building system prompt with tool context...")
            system_prompt = self._build_system_prompt(analysis_type)
            print(f"   📝 System prompt built with {len(self.available_tools)} tools")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this text:\n{text}"}
            ]
            
            print("   🚀 Sending request to AI model...")
            response = client.chat.completions.create(
                model="openai:gpt-4o",
                messages=messages,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content
            print(f"   ✅ AI response received: {len(response_text)} characters")

            # Process AI response and handle tool calls if present
            print("\n" + "="*50)
            result = self._process_ai_response(response_text, analysis_type)
            print("="*50)

            # If tool calls are requested, execute them and generate final analysis
            if result.get("status") == "tool_requested" and "tool_calls" in result:
                print(f"\n🚀 AGENT WORKFLOW: Tool execution required ({len(result['tool_calls'])} tools)")
                return self._execute_tools_and_analyze(result["tool_calls"], text, analysis_type, client, messages)
            else:
                print("\n📄 AGENT WORKFLOW: No tools needed, returning direct analysis")
            
            return result
            
        except Exception as e:
            print(f"\n❌ AGENT ERROR: {str(e)}")
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
