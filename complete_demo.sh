#!/bin/bash

echo "🎬 COMPLETE AGENT TOOL SELECTION & EXECUTION DEMO"
echo "=" * 60
echo ""

echo "📋 1. SINGLE TOOL - MULTIPLY:"
echo "Command: Calculate 6 times 7"
echo ""
.venv/bin/python agent.py '{"method": "analyze_text", "parameters": {"text": "Calculate 6 times 7", "analysis_type": "mathematical"}, "tool_context": {"available_tools": ["multiply"], "tool_descriptions": {"multiply": "Multiply two numbers"}, "tool_usage_examples": {"multiply": ["multiply({\"a\": \"6\", \"b\": \"7\"})"]}}}'

echo ""
echo "=" * 60
echo ""

echo "📋 2. SINGLE TOOL - ADD:"
echo "Command: Add 60 and 8"
echo ""
.venv/bin/python agent.py '{"method": "analyze_text", "parameters": {"text": "Add 60 and 8", "analysis_type": "mathematical"}, "tool_context": {"available_tools": ["add"], "tool_descriptions": {"add": "Add two numbers"}, "tool_usage_examples": {"add": ["add({\"a\": \"60\", \"b\": \"8\"})"]}}}'

echo ""
echo "=" * 60
echo ""

echo "📋 3. MULTI-STEP OPERATION:"
echo "Command: Calculate 12 times 5, then add 8"
echo "(Shows first step - multiply tool execution)"
echo ""
.venv/bin/python agent.py '{"method": "analyze_text", "parameters": {"text": "Calculate 12 times 5, then add 8 to the result", "analysis_type": "mathematical"}, "tool_context": {"available_tools": ["multiply", "add"], "tool_descriptions": {"multiply": "Multiply two numbers", "add": "Add two numbers"}, "tool_usage_examples": {"multiply": ["multiply({\"a\": \"12\", \"b\": \"5\"})"], "add": ["add({\"a\": \"60\", \"b\": \"8\"})"]}}}'

echo ""
echo "🎉 COMPLETE DEMONSTRATION FINISHED!"
echo "All tool selection and execution steps are now visible!"
