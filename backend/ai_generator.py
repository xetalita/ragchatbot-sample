import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt template to avoid rebuilding on each call
    SYSTEM_PROMPT_TEMPLATE = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Tool Usage Guidelines:
- **Course content/material questions**: Use search_course_content for specific educational content
- **Course outline/structure questions**: Use get_course_outline for course title, course link, and lesson listings
- **Sequential tool strategy**: You can make up to {max_rounds} tool calls across separate rounds
- **Strategic usage**: Use get_course_outline first to understand structure, then search_course_content with specific targets
- **Early termination**: Provide final answer when sufficient information is gathered
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Available Tools:
1. **search_course_content**: For searching specific course content and materials
2. **get_course_outline**: For retrieving course structure (title, course link, complete lesson list with numbers and titles)

Sequential Tool Examples:
- Query: "Find courses similar to lesson 4 of course X"
- Round 1: get_course_outline(course_name="course X") → get lesson 4 details
- Round 2: search_course_content(query="lesson 4 topic") → find similar content

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course content questions**: Use search_course_content first, then answer
- **Course outline/syllabus questions**: Use get_course_outline first, then answer
- **Complex queries**: Use multiple rounds strategically to gather comprehensive information
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the outline tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         max_rounds: int = 2) -> str:
        """
        Generate AI response with optional sequential tool usage and conversation context.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum sequential tool calling rounds (default 2)
            
        Returns:
            Generated response as string
        """
        
        # Build system content with dynamic max_rounds
        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(max_rounds=max_rounds)
        system_content = (
            f"{system_prompt}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else system_prompt
        )
        
        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}
        
        # Get response from Claude
        response = self.client.messages.create(**api_params)
        
        # Handle tool execution - sequential if tools available, single if not
        if response.stop_reason == "tool_use" and tool_manager:
            if tools and max_rounds > 1:
                return self._execute_sequential_rounds(query, system_content, tools, tool_manager, max_rounds, response)
            else:
                return self._handle_tool_execution(response, api_params, tool_manager)
        
        # Return direct response
        return response.content[0].text
    
    def _execute_sequential_rounds(self, query: str, system_content: str, 
                                  tools: List, tool_manager, max_rounds: int, initial_response=None) -> str:
        """
        Execute sequential tool calling rounds with conversation context accumulation.
        
        Args:
            query: The user's original question
            system_content: System prompt with conversation history
            tools: Available tools for all rounds
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of rounds to execute
            initial_response: Optional initial response that already has tool calls
            
        Returns:
            Final response after sequential tool execution
        """
        messages = [{"role": "user", "content": query}]
        
        # If we have an initial response with tools, process it first
        if initial_response and initial_response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": initial_response.content})
            
            # Execute tools from initial response
            tool_results = []
            for content_block in initial_response.content:
                if content_block.type == "tool_use":
                    tool_result = tool_manager.execute_tool(
                        content_block.name,
                        **content_block.input
                    )
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result
                    })
            
            # Add tool results to conversation
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            
            # We've used one round already, so reduce remaining rounds
            remaining_rounds = max_rounds - 1
        else:
            remaining_rounds = max_rounds
        
        for _ in range(remaining_rounds):
            # Build API parameters for this round
            api_params = {
                **self.base_params,
                "messages": messages.copy(),
                "system": system_content,
                "tools": tools,
                "tool_choice": {"type": "auto"}
            }
            
            # Make API call for this round
            response = self.client.messages.create(**api_params)
            
            # Add AI's response to conversation
            messages.append({"role": "assistant", "content": response.content})
            
            # Check if AI used tools
            if response.stop_reason == "tool_use":
                # Execute tools for this round
                tool_results = []
                for content_block in response.content:
                    if content_block.type == "tool_use":
                        tool_result = tool_manager.execute_tool(
                            content_block.name,
                            **content_block.input
                        )
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result
                        })
                
                # Add tool results to conversation if any were executed
                if tool_results:
                    messages.append({"role": "user", "content": tool_results})
            else:
                # AI provided final answer without tools - early termination
                return response.content[0].text
        
        # Exhausted all rounds - make final synthesis call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": f"{system_content}\n\nSYNTHESIS: Provide final answer based on all gathered information."
        }
        
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call with tools still available for potential sequential use
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
            "tools": base_params.get("tools"),
            "tool_choice": base_params.get("tool_choice")
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text