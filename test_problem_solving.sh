#!/bin/bash

# Test script for the new problem-solving capabilities
echo "🧩 Testing Agent Problem Solving Capabilities"
echo "============================================="

# Test 1: Math problem with multiple steps
echo -e "\n📐 Test 1: Multi-step Math Problem"
echo "Problem: Calculate the area of a rectangle with width 12 and height 8, then find what percentage this area is of a square with side 15."

python3 agent.py '{
    "method": "solve_problem",
    "parameters": {
        "problem": "Calculate the area of a rectangle with width 12 and height 8, then find what percentage this area is of a square with side 15.",
        "context": "Show all calculation steps"
    },
    "tool_context": {
        "available_tools": ["multiply", "divide"],
        "tool_descriptions": {
            "multiply": "Multiplies two numbers",
            "divide": "Divides two numbers"
        },
        "tool_usage_examples": {
            "multiply": ["12 * 8", "area calculation"],
            "divide": ["96 / 225", "percentage calculation"]
        }
    }
}'

echo -e "\n" && sleep 2

# Test 2: Text analysis problem
echo -e "\n📝 Test 2: Complex Text Analysis Problem"
echo "Problem: Analyze the sentiment of a product review, extract key points, and provide improvement suggestions."

python3 agent.py '{
    "method": "solve_problem", 
    "parameters": {
        "problem": "Analyze the sentiment of this product review, extract key points, and provide improvement suggestions",
        "context": "Review: The smartphone has excellent camera quality and fast performance, but the battery life is disappointing and it gets warm during heavy use. The price is reasonable for the features offered."
    },
    "tool_context": {
        "available_tools": ["text_analyzer", "sentiment_tool"],
        "tool_descriptions": {
            "text_analyzer": "Analyzes text content and extracts insights",
            "sentiment_tool": "Determines sentiment of text"
        },
        "tool_usage_examples": {
            "text_analyzer": ["extract key points", "analyze content"],
            "sentiment_tool": ["positive/negative detection", "emotion analysis"]
        }
    }
}'

echo -e "\n" && sleep 2

# Test 3: Data processing problem
echo -e "\n📊 Test 3: Data Processing Problem"
echo "Problem: Given sales data for Q1-Q4: 150, 200, 180, 220, calculate the total annual sales, average quarterly sales, and growth rate from Q1 to Q4."

python3 agent.py '{
    "method": "solve_problem",
    "parameters": {
        "problem": "Given sales data for Q1-Q4: 150, 200, 180, 220, calculate the total annual sales, average quarterly sales, and growth rate from Q1 to Q4.",
        "context": "Sales figures are in thousands of dollars"
    },
    "tool_context": {
        "available_tools": ["add", "divide", "subtract"],
        "tool_descriptions": {
            "add": "Adds multiple numbers",
            "divide": "Divides two numbers", 
            "subtract": "Subtracts two numbers"
        },
        "tool_usage_examples": {
            "add": ["sum quarterly sales", "total calculation"],
            "divide": ["average calculation", "percentage calculation"],
            "subtract": ["growth calculation", "difference calculation"]
        }
    }
}'

echo -e "\n📋 Problem Solving Tests Complete!"
