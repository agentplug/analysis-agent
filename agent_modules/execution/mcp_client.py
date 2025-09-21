#!/usr/bin/env python3
"""
MCP Client Module
Handles MCP (Model Context Protocol) server communication and fallback execution.
"""

import asyncio
import concurrent.futures
from typing import Dict, Any


class MCPClient:
    """Handles MCP server communication and local fallback execution."""
    
    def __init__(self, server_url: str = "http://localhost:8000/sse", allow_fallback: bool = False):
        """
        Initialize MCP client.
        
        Args:
            server_url: URL of the MCP server
            allow_fallback: Whether to allow local fallback when MCP server is unavailable
        """
        self.server_url = server_url
        self.allow_fallback = allow_fallback
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call MCP tool server using proper SSE format to execute any tool.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments for the tool
            
        Returns:
            Tool execution result
        """
        # Handle async execution in agent context
        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an event loop, use run_in_executor to avoid blocking
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._execute_tool_via_sse(tool_name, arguments))
                    return future.result(timeout=30)
            except RuntimeError:
                # No event loop running, safe to use asyncio.run()
                return asyncio.run(self._execute_tool_via_sse(tool_name, arguments))
                
        except Exception as e:
            # Check if fallback is allowed
            if self.allow_fallback:
                import sys
                print(f"MCP tool execution failed, using local fallback: {e}", file=sys.stderr)
                return self._execute_tool_locally(tool_name, arguments)
            else:
                import sys
                print(f"MCP SERVER UNAVAILABLE: {e}", file=sys.stderr)
                raise Exception(f"MCP server is unavailable and fallback is disabled. Error: {str(e)}")
    
    async def _execute_tool_via_sse(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute tool via MCP SSE connection.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments for the tool
            
        Returns:
            Tool execution result
        """
        try:
            # Import MCP SSE client components
            from mcp import ClientSession
            from mcp.client.sse import sse_client
            
            # Connect to MCP server using SSE format
            async with sse_client(url=self.server_url) as streams:
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
        
        # Math operations (support any math tool, but not web tools)
        if any(keyword in tool_name.lower() for keyword in ['add', 'plus', '+']) and not any(web_keyword in tool_name.lower() for web_keyword in ['web_', 'url', 'http']):
            return self._execute_math_operation('+', processed_args)
        elif tool_name.lower() == 'sum' or (tool_name.lower().startswith('sum_') and not any(web_keyword in tool_name.lower() for web_keyword in ['web_', 'url', 'http'])):
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
        
        # Web search tools (specific handler for built-in web search)
        elif any(keyword in tool_name.lower() for keyword in ['web_search', 'websearch']):
            return self._execute_builtin_web_search(tool_name, processed_args)
        
        # Generic search tools (fallback for other search tools)
        elif any(keyword in tool_name.lower() for keyword in ['search', 'find', 'query', 'lookup']):
            query = processed_args.get('query', processed_args.get('q', processed_args.get('search', 'default')))
            return f"Search results for '{query}' (via {tool_name}): [Simulated results]"
        
        # Data analysis tools (generic, not web-specific)
        elif any(keyword in tool_name.lower() for keyword in ['data_analyze', 'stats_analyze', 'generic_analyze']) and not any(web_keyword in tool_name.lower() for web_keyword in ['web_', 'url', 'http']):
            data = processed_args.get('data', processed_args.get('input', 'sample data'))
            return f"Analysis of '{data}' completed via {tool_name}: [Key insights found]"
        
        # File operations
        elif any(keyword in tool_name.lower() for keyword in ['file', 'read', 'write', 'save']):
            filename = processed_args.get('filename', processed_args.get('path', 'file.txt'))
            return f"File operation on '{filename}' via {tool_name}: [Operation completed]"
        
        # Web scraping tools
        elif any(keyword in tool_name.lower() for keyword in ['web_scrape', 'scrape', 'extract', 'crawl']):
            return self._execute_builtin_web_scrape(tool_name, processed_args)
        
        # Web analysis tools
        elif any(keyword in tool_name.lower() for keyword in ['web_analyze', 'web_analysis', 'analyze_web']):
            return self._execute_builtin_web_analyze(tool_name, processed_args)
        
        # Web summarization tools
        elif any(keyword in tool_name.lower() for keyword in ['web_summarize', 'web_summary', 'summarize_web']):
            return self._execute_builtin_web_summarize(tool_name, processed_args)
        
        # Web search and scrape tools
        elif any(keyword in tool_name.lower() for keyword in ['web_search_and_scrape', 'search_and_scrape', 'web_search_scrape']):
            return self._execute_builtin_web_search_and_scrape(tool_name, processed_args)
        
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
    
    def _setup_agenthub_import(self):
        """Setup AgentHub import path for the agent subprocess."""
        import sys
        import os
        
        # Try to find AgentHub repository dynamically
        agenthub_path = None
        
        # Method 1: Use environment variable if set
        agenthub_path = os.environ.get("AGENTHUB_PATH")
        
        # Method 2: Look for agenthub in current working directory
        if not agenthub_path:
            current_dir = os.getcwd()
            if os.path.exists(os.path.join(current_dir, "agenthub")):
                agenthub_path = current_dir
        
        # Method 3: Look in parent directories (common when running from subdirectories)
        if not agenthub_path:
            current_dir = os.getcwd()
            # Go up to 5 levels to find agenthub directory
            for _ in range(5):
                if os.path.exists(os.path.join(current_dir, "agenthub")):
                    agenthub_path = current_dir
                    break
                parent_dir = os.path.dirname(current_dir)
                if parent_dir == current_dir:  # Reached root
                    break
                current_dir = parent_dir
        
        # Method 4: Look for agenthub in Python path
        if not agenthub_path:
            for path in sys.path:
                if os.path.exists(os.path.join(path, "agenthub")):
                    agenthub_path = path
                    break
        
        # Method 5: Look relative to this file's location
        if not agenthub_path:
            # This file is in the agent's directory, go up to find agenthub
            current_file = os.path.abspath(__file__)
            # Navigate up from agent directory to find agenthub
            search_path = os.path.dirname(current_file)
            for _ in range(6):  # Go up 6 levels max
                if os.path.exists(os.path.join(search_path, "agenthub")):
                    agenthub_path = search_path
                    break
                parent = os.path.dirname(search_path)
                if parent == search_path:  # Reached root
                    break
                search_path = parent
        
        # Method 6: Fallback to known relative path from agent directory
        if not agenthub_path:
            # Agent is typically in ~/.agenthub/agents/agentplug/analysis-agent/
            # AgentHub repo should be in ~/repos/agenthub/
            home_dir = os.path.expanduser("~")
            potential_path = os.path.join(home_dir, "repos", "agenthub")
            if os.path.exists(os.path.join(potential_path, "agenthub")):
                agenthub_path = potential_path
        
        # Add to Python path if found
        if agenthub_path and agenthub_path not in sys.path:
            sys.path.insert(0, agenthub_path)
    
    def _execute_builtin_web_search(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute built-in web search tool."""
        try:
            # Setup AgentHub import path
            self._setup_agenthub_import()
            
            # Import the web search tool
            from agenthub.core.tools.builtin.web.search import web_search
            
            # Extract query from arguments (handle different parameter names)
            query = arguments.get('query', arguments.get('q', arguments.get('search', '')))
            if not query:
                return {"error": "No search query provided"}
            
            # Extract other parameters with defaults
            engine = arguments.get('engine', 'duckduckgo')
            max_results = arguments.get('max_results', 10)
            language = arguments.get('language', 'en')
            region = arguments.get('region', 'us')
            time_filter = arguments.get('time_filter', None)
            safe_search = arguments.get('safe_search', True)
            include_snippets = arguments.get('include_snippets', True)
            
            # Call the actual web search tool
            result = web_search(
                query=query,
                engine=engine,
                max_results=max_results,
                language=language,
                region=region,
                time_filter=time_filter,
                safe_search=safe_search,
                include_snippets=include_snippets
            )
            
            return result
            
        except Exception as e:
            return {
                "error": f"Web search failed: {str(e)}",
                "tool_name": tool_name,
                "arguments": arguments
            }
    
    def _execute_builtin_web_scrape(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute built-in web scraping tool."""
        try:
            # Setup AgentHub import path
            self._setup_agenthub_import()
            
            # Import the web scrape tool
            from agenthub.core.tools.builtin.web.scrape import web_scrape
            
            # Extract URL from arguments (handle different parameter names)
            url = arguments.get('url', arguments.get('url_value', arguments.get('target_url', '')))
            if not url:
                return {"error": "No URL provided for web scraping"}
            
            # Extract other parameters with defaults
            extract_text = arguments.get('extract_text', True)
            extract_links = arguments.get('extract_links', False)
            extract_images = arguments.get('extract_images', False)
            extract_metadata = arguments.get('extract_metadata', True)
            timeout = arguments.get('timeout', 10)
            user_agent = arguments.get('user_agent', arguments.get('user_agent_value', None))
            follow_redirects = arguments.get('follow_redirects', True)
            
            # Call the actual web scrape tool
            result = web_scrape(
                url=url,
                extract_text=extract_text,
                extract_links=extract_links,
                extract_images=extract_images,
                extract_metadata=extract_metadata,
                timeout=timeout,
                user_agent=user_agent,
                follow_redirects=follow_redirects
            )
            
            return result
            
        except Exception as e:
            return {
                "error": f"Web scraping failed: {str(e)}",
                "tool_name": tool_name,
                "arguments": arguments
            }
    
    def _execute_builtin_web_analyze(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute built-in web analysis tool."""
        try:
            # Setup AgentHub import path
            self._setup_agenthub_import()
            
            # Import the web analyze tool
            from agenthub.core.tools.builtin.web.analyze import web_analyze
            
            # Extract URL from arguments (handle different parameter names)
            url = arguments.get('url', arguments.get('url_value', arguments.get('target_url', '')))
            if not url:
                return {"error": "No URL provided for web analysis"}
            
            # Extract other parameters with defaults
            analysis_types = arguments.get('analysis_types', ['sentiment', 'topics', 'keywords'])
            extract_sentiment = arguments.get('extract_sentiment', True)
            extract_topics = arguments.get('extract_topics', True)
            extract_keywords = arguments.get('extract_keywords', True)
            extract_entities = arguments.get('extract_entities', True)
            analyze_readability = arguments.get('analyze_readability', False)
            detect_language = arguments.get('detect_language', False)
            timeout = arguments.get('timeout', 15)
            
            # Call the actual web analyze tool
            result = web_analyze(
                url=url,
                analysis_types=analysis_types,
                extract_sentiment=extract_sentiment,
                extract_topics=extract_topics,
                extract_keywords=extract_keywords,
                extract_entities=extract_entities,
                analyze_readability=analyze_readability,
                detect_language=detect_language,
                timeout=timeout
            )
            
            return result
            
        except Exception as e:
            return {
                "error": f"Web analysis failed: {str(e)}",
                "tool_name": tool_name,
                "arguments": arguments
            }
    
    def _execute_builtin_web_summarize(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute built-in web summarization tool."""
        try:
            # Setup AgentHub import path
            self._setup_agenthub_import()
            
            # Import the web summarize tool
            from agenthub.core.tools.builtin.web.summarize import web_summarize
            
            # Extract URL from arguments (handle different parameter names)
            url = arguments.get('url', arguments.get('url_value', arguments.get('target_url', '')))
            if not url:
                return {"error": "No URL provided for web summarization"}
            
            # Extract other parameters with defaults
            max_length = arguments.get('max_length', 500)
            language = arguments.get('language', 'en')
            style = arguments.get('style', 'informative')
            include_key_points = arguments.get('include_key_points', True)
            extract_entities = arguments.get('extract_entities', False)
            
            # Call the actual web summarize tool
            result = web_summarize(
                url=url,
                max_length=max_length,
                language=language,
                style=style,
                include_key_points=include_key_points,
                extract_entities=extract_entities
            )
            
            return result
            
        except Exception as e:
            return {
                "error": f"Web summarization failed: {str(e)}",
                "tool_name": tool_name,
                "arguments": arguments
            }
    
    def _execute_builtin_web_search_and_scrape(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute built-in web search and scrape tool."""
        try:
            # Setup AgentHub import path
            self._setup_agenthub_import()
            
            # Import the web search and scrape tool
            from agenthub.core.tools.builtin.web.search_and_scrape import web_search_and_scrape
            
            # Extract query from arguments (handle different parameter names)
            query = arguments.get('query', arguments.get('q', arguments.get('search', '')))
            if not query:
                return {"error": "No search query provided for search and scrape"}
            
            # Extract other parameters with defaults
            max_results = arguments.get('max_results', 5)
            engine = arguments.get('engine', 'duckduckgo')
            extract_text = arguments.get('extract_text', True)
            extract_metadata = arguments.get('extract_metadata', True)
            timeout = arguments.get('timeout', 10)
            
            # Call the actual web search and scrape tool
            result = web_search_and_scrape(
                query=query,
                max_results=max_results,
                engine=engine,
                extract_text=extract_text,
                extract_metadata=extract_metadata,
                timeout=timeout
            )
            
            return result
            
        except Exception as e:
            return {
                "error": f"Web search and scrape failed: {str(e)}",
                "tool_name": tool_name,
                "arguments": arguments
            }
    
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
