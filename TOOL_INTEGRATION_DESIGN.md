# Tool Integration Design for Analysis Agent

## Overview

This document outlines the design for integrating dynamic tool injection capabilities into the analysis-agent while maintaining minimal changes to the existing implementation. The goal is to make the agent tool-agnostic and capable of using any tools provided by the AgentHub framework.

## Current Architecture

The current analysis-agent has a simple, clean architecture:
- `AnalysisAgent` class with `analyze_text()` and `summarize_content()` methods
- Command-line interface via `main()` function
- JSON-based input/output communication
- AI-powered analysis using aisuite

## Tool Integration Requirements

### 1. Tool Context Injection
The agent needs to receive tool metadata and context information to understand what tools are available and how to use them.

**Input Format**:
```json
{
  "method": "analyze_text",
  "parameters": {
    "text": "Text to analyze",
    "analysis_type": "general"
  },
  "tool_context": {
    "available_tools": ["web_search", "data_analyzer", "custom_tool"],
    "tool_descriptions": {
      "web_search": "Search the web for real-time information",
      "data_analyzer": "Analyze data and provide insights",
      "custom_tool": "Custom tool description"
    },
    "tool_usage_examples": {
      "web_search": ["web_search(query=\"AI trends\", max_results=5)"],
      "data_analyzer": ["data_analyzer(data=\"text to analyze\")"]
    }
  }
}
```

### 2. Tool Call Detection
The agent needs to detect when it wants to use a tool and format the request properly.

**Tool Call Format**:
```json
{
  "tool_call": {
    "tool_name": "web_search",
    "arguments": {
      "query": "latest AI trends",
      "max_results": 3
    }
  },
  "analysis": "I'll search for current information to enhance my analysis."
}
```

### 3. Tool Execution Interface
The agent needs a way to execute tools and receive results.

**Tool Execution Request**:
```json
{
  "action": "execute_tool",
  "tool_name": "web_search",
  "arguments": {
    "query": "latest AI trends",
    "max_results": 3
  }
}
```

**Tool Execution Response**:
```json
{
  "result": {
    "query": "latest AI trends",
    "results": [
      {
        "title": "AI Trends 2024",
        "snippet": "Latest developments in AI...",
        "url": "https://example.com"
      }
    ]
  },
  "success": true
}
```

## Implementation Design

### 1. Minimal Changes to AnalysisAgent Class

**Add tool context support**:
```python
class AnalysisAgent:
    def __init__(self, tool_context=None):
        """Initialize the analysis agent with optional tool context."""
        self.tool_context = tool_context or {}
        self.available_tools = self.tool_context.get("available_tools", [])
        self.tool_descriptions = self.tool_context.get("tool_descriptions", {})
        self.tool_usage_examples = self.tool_context.get("tool_usage_examples", {})
```

**Enhance system prompt generation**:
```python
def _build_system_prompt(self, analysis_type: str) -> str:
    """Build system prompt with tool context if available."""
    base_prompts = {
        "sentiment": "You are a sentiment analyzer...",
        "key_points": "You are a key points extractor...",
        "summary": "You are a text summarizer...",
        "general": "You are a comprehensive text analyzer..."
    }
    
    base_prompt = base_prompts.get(analysis_type, base_prompts["general"])
    
    if self.available_tools:
        tool_context = self._build_tool_context_string()
        return f"{base_prompt}\n\n{tool_context}"
    
    return base_prompt

def _build_tool_context_string(self) -> str:
    """Build tool context string for AI system prompt."""
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
```

### 2. Tool Call Processing

**Add tool call detection and processing**:
```python
def _process_ai_response(self, response: str, analysis_type: str) -> Dict[str, Any]:
    """Process AI response and handle tool calls if present."""
    # Try to detect tool calls in the response
    tool_calls = self._extract_tool_calls(response)
    
    if tool_calls:
        return {
            "tool_calls": tool_calls,
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
    """Extract tool calls from AI response."""
    tool_calls = []
    
    # Look for JSON tool calls in the response
    import re
    json_pattern = r'\{[^{}]*"tool_call"[^{}]*\}'
    matches = re.findall(json_pattern, response, re.DOTALL)
    
    for match in matches:
        try:
            tool_call = json.loads(match)
            if "tool_call" in tool_call:
                tool_calls.append(tool_call["tool_call"])
        except json.JSONDecodeError:
            continue
    
    return tool_calls
```

### 3. Enhanced Main Function

**Modify main function to handle tool context**:
```python
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
```

## Tool Execution Flow

### 1. Agent Receives Tool Context
The AgentHub framework injects tool metadata into the agent's input.

### 2. Agent Decides to Use Tools
The AI analyzes the input and decides whether tools would be helpful.

### 3. Agent Requests Tool Execution
The agent returns a tool call request instead of a normal analysis result.

### 4. Framework Executes Tools
The AgentHub framework executes the requested tools and returns results.

### 5. Agent Processes Tool Results
The agent receives tool results and incorporates them into the final analysis.

## Example Usage Scenarios

### Scenario 1: Web Search for Current Information
```json
// Input
{
  "method": "analyze_text",
  "parameters": {
    "text": "What are the latest trends in AI?",
    "analysis_type": "general"
  },
  "tool_context": {
    "available_tools": ["web_search"],
    "tool_descriptions": {
      "web_search": "Search the web for real-time information"
    }
  }
}

// Agent Response (Tool Call Request)
{
  "result": {
    "tool_calls": [
      {
        "tool_name": "web_search",
        "arguments": {
          "query": "latest AI trends 2024",
          "max_results": 5
        }
      }
    ],
    "status": "tool_requested"
  }
}
```

### Scenario 2: Data Analysis for Content
```json
// Input
{
  "method": "analyze_text",
  "parameters": {
    "text": "This product is amazing! I love it so much. The quality is excellent.",
    "analysis_type": "sentiment"
  },
  "tool_context": {
    "available_tools": ["data_analyzer"],
    "tool_descriptions": {
      "data_analyzer": "Analyze data and provide insights"
    }
  }
}

// Agent Response (Tool Call Request)
{
  "result": {
    "tool_calls": [
      {
        "tool_name": "data_analyzer",
        "arguments": {
          "data": "This product is amazing! I love it so much. The quality is excellent."
        }
      }
    ],
    "status": "tool_requested"
  }
}
```

### Scenario 3: No Tools Needed
```json
// Input
{
  "method": "analyze_text",
  "parameters": {
    "text": "Simple text analysis",
    "analysis_type": "general"
  },
  "tool_context": {
    "available_tools": ["web_search", "data_analyzer"]
  }
}

// Agent Response (Normal Analysis)
{
  "result": {
    "main_themes": ["text analysis"],
    "tone": "neutral",
    "structure": "simple",
    "key_insights": ["Basic text analysis"],
    "summary": "Simple text requiring basic analysis"
  }
}
```

## Benefits of This Design

### 1. Minimal Changes
- Only adds tool context support to the constructor
- Enhances existing system prompt generation
- Adds tool call detection and processing
- Maintains backward compatibility

### 2. Tool Agnostic
- Works with any tools provided by the framework
- No hardcoded tool dependencies
- Flexible tool context format

### 3. Backward Compatible
- Existing functionality remains unchanged
- Works without tool context
- Graceful degradation

### 4. Extensible
- Easy to add new tool types
- Supports complex tool interactions
- Framework handles tool execution complexity

## Implementation Steps

1. **Add tool context support** to `AnalysisAgent.__init__()`
2. **Enhance system prompt generation** with tool context
3. **Add tool call detection** in response processing
4. **Modify main function** to handle tool context
5. **Test with various tool combinations**

## Testing Strategy

### Unit Tests
- Test tool context parsing
- Test system prompt generation with tools
- Test tool call detection
- Test backward compatibility

### Integration Tests
- Test with real tools (web_search, data_analyzer)
- Test tool execution flow
- Test error handling
- Test performance

### End-to-End Tests
- Test complete tool injection workflow
- Test with multiple tool types
- Test with no tools
- Test with invalid tools

## Conclusion

This design provides a clean, minimal way to add tool injection capabilities to the analysis-agent while maintaining its existing functionality and architecture. The agent becomes tool-aware but remains tool-agnostic, allowing the AgentHub framework to provide any tools dynamically.
