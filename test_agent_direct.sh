#!/bin/bash

echo "🎯 TESTING AGENT DIRECTLY WITH FULL LOGGING"
echo "============================================"

cd /Users/aitomatic/.agenthub/agents/agentplug/analysis-agent

echo "Running agent with multiply tool..."
echo ""

.venv/bin/python agent.py '{
  "method": "analyze_text", 
  "parameters": {
    "text": "Calculate 8 times 9 using multiply tool", 
    "analysis_type": "mathematical"
  }, 
  "tool_context": {
    "available_tools": ["multiply"], 
    "tool_descriptions": {
      "multiply": "Multiply two numbers together"
    }, 
    "tool_usage_examples": {
      "multiply": ["multiply({\"a\": \"8\", \"b\": \"9\"})"]
    }
  }
}'
