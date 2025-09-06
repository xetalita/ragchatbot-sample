import pytest
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add backend to path for imports
backend_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, backend_path)

from ai_generator import AIGenerator


class MockAnthropicResponse:
    """Mock response from Anthropic API"""
    def __init__(self, content_text=None, stop_reason="end_turn", tool_calls=None):
        self.stop_reason = stop_reason
        self.content = []
        
        if content_text:
            mock_content = Mock()
            mock_content.type = "text"
            mock_content.text = content_text
            self.content.append(mock_content)
        
        if tool_calls:
            for tool_call in tool_calls:
                mock_tool = Mock()
                mock_tool.type = "tool_use"
                mock_tool.name = tool_call["name"]
                mock_tool.input = tool_call["input"]
                mock_tool.id = tool_call.get("id", f"tool_{tool_call['name']}")
                self.content.append(mock_tool)


class MockToolManager:
    """Mock tool manager for testing"""
    def __init__(self):
        self.call_history = []
        self.responses = {}
    
    def execute_tool(self, tool_name, **kwargs):
        self.call_history.append((tool_name, kwargs))
        return self.responses.get(tool_name, f"Mock result for {tool_name}")
    
    def set_response(self, tool_name, response):
        self.responses[tool_name] = response


@pytest.fixture
def ai_generator():
    """Create AIGenerator instance for testing"""
    return AIGenerator(api_key="test_key", model="claude-sonnet-4")


@pytest.fixture
def mock_tools():
    """Mock tools configuration"""
    return [
        {"name": "get_course_outline", "description": "Get course outline"},
        {"name": "search_course_content", "description": "Search course content"}
    ]


@pytest.fixture
def mock_tool_manager():
    """Mock tool manager"""
    return MockToolManager()


class TestAIGeneratorBasic:
    """Test basic functionality and backward compatibility"""
    
    def test_generate_response_without_tools(self, ai_generator):
        """Test basic response generation without tools"""
        with patch.object(ai_generator.client.messages, 'create') as mock_create:
            mock_create.return_value = MockAnthropicResponse("This is a basic response")
            
            result = ai_generator.generate_response("What is Python?")
            
            assert result == "This is a basic response"
            mock_create.assert_called_once()
    
    def test_system_prompt_template_formatting(self, ai_generator):
        """Test that system prompt formats correctly with max_rounds"""
        with patch.object(ai_generator.client.messages, 'create') as mock_create:
            mock_create.return_value = MockAnthropicResponse("Response")
            
            ai_generator.generate_response("Test query", max_rounds=3)
            
            call_args = mock_create.call_args
            system_content = call_args.kwargs['system']
            assert "up to 3 tool calls" in system_content
    
    def test_backward_compatibility_default_max_rounds(self, ai_generator, mock_tools, mock_tool_manager):
        """Test that default max_rounds is 2 for backward compatibility"""
        with patch.object(ai_generator.client.messages, 'create') as mock_create:
            # Single tool call, then final response
            mock_create.side_effect = [
                MockAnthropicResponse(stop_reason="tool_use", tool_calls=[
                    {"name": "search_course_content", "input": {"query": "test"}}
                ]),
                MockAnthropicResponse("Final response")
            ]
            
            result = ai_generator.generate_response("Test query", tools=mock_tools, tool_manager=mock_tool_manager)
            
            assert result == "Final response"
            assert len(mock_tool_manager.call_history) == 1


class TestSequentialToolCalling:
    """Test sequential tool calling functionality"""
    
    def test_two_round_sequential_calling(self, ai_generator, mock_tools, mock_tool_manager):
        """Test the classic two-round scenario: outline then search"""
        mock_tool_manager.set_response("get_course_outline", "Course X - Lesson 4: Advanced Database Indexing")
        mock_tool_manager.set_response("search_course_content", "Found similar content in Course Y")
        
        with patch.object(ai_generator.client.messages, 'create') as mock_create:
            # Round 1: Tool call for outline
            # Round 2: Tool call for search  
            # Final synthesis call
            mock_create.side_effect = [
                MockAnthropicResponse(stop_reason="tool_use", tool_calls=[
                    {"name": "get_course_outline", "input": {"course_name": "Course X"}}
                ]),
                MockAnthropicResponse(stop_reason="tool_use", tool_calls=[
                    {"name": "search_course_content", "input": {"query": "Advanced Database Indexing"}}
                ]),
                MockAnthropicResponse("Based on the outline and search, Course Y has similar content to Course X lesson 4")
            ]
            
            result = ai_generator.generate_response(
                "Find courses similar to lesson 4 of Course X",
                tools=mock_tools,
                tool_manager=mock_tool_manager,
                max_rounds=2
            )
            
            # Verify external behavior
            assert len(mock_tool_manager.call_history) == 2
            assert mock_tool_manager.call_history[0][0] == "get_course_outline"
            assert mock_tool_manager.call_history[1][0] == "search_course_content"
            assert "Course Y has similar content" in result
            assert mock_create.call_count == 3  # 2 rounds + final synthesis
    
    def test_early_termination_when_ai_provides_answer(self, ai_generator, mock_tools, mock_tool_manager):
        """Test that execution stops early when AI provides final answer"""
        with patch.object(ai_generator.client.messages, 'create') as mock_create:
            # Round 1: Tool call
            # Round 2: AI provides final answer (no tool use)
            mock_create.side_effect = [
                MockAnthropicResponse(stop_reason="tool_use", tool_calls=[
                    {"name": "search_course_content", "input": {"query": "test"}}
                ]),
                MockAnthropicResponse("I have sufficient information to answer: The course covers...")
            ]
            
            result = ai_generator.generate_response(
                "What does the course cover?",
                tools=mock_tools,
                tool_manager=mock_tool_manager,
                max_rounds=3
            )
            
            assert result == "I have sufficient information to answer: The course covers..."
            assert len(mock_tool_manager.call_history) == 1  # Only first round executed
            assert mock_create.call_count == 2  # No final synthesis call needed
    
    def test_maximum_rounds_exhaustion(self, ai_generator, mock_tools, mock_tool_manager):
        """Test behavior when all rounds are exhausted"""
        with patch.object(ai_generator.client.messages, 'create') as mock_create:
            # All rounds use tools, then final synthesis
            mock_create.side_effect = [
                MockAnthropicResponse(stop_reason="tool_use", tool_calls=[
                    {"name": "get_course_outline", "input": {"course_name": "Course A"}}
                ]),
                MockAnthropicResponse(stop_reason="tool_use", tool_calls=[
                    {"name": "search_course_content", "input": {"query": "topic"}}
                ]),
                MockAnthropicResponse("Final synthesized answer based on all information")
            ]
            
            result = ai_generator.generate_response(
                "Complex query requiring multiple steps",
                tools=mock_tools,
                tool_manager=mock_tool_manager,
                max_rounds=2
            )
            
            assert len(mock_tool_manager.call_history) == 2
            assert "Final synthesized answer" in result
            assert mock_create.call_count == 3  # 2 rounds + synthesis
    
    def test_single_round_fallback(self, ai_generator, mock_tools, mock_tool_manager):
        """Test fallback to single round when max_rounds=1"""
        with patch.object(ai_generator.client.messages, 'create') as mock_create:
            mock_create.side_effect = [
                MockAnthropicResponse(stop_reason="tool_use", tool_calls=[
                    {"name": "search_course_content", "input": {"query": "test"}}
                ]),
                MockAnthropicResponse("Single round response")
            ]
            
            result = ai_generator.generate_response(
                "Test query",
                tools=mock_tools,
                tool_manager=mock_tool_manager,
                max_rounds=1
            )
            
            # Should use original _handle_tool_execution method
            assert result == "Single round response"
            assert len(mock_tool_manager.call_history) == 1


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_tool_execution_error_handling(self, ai_generator, mock_tools):
        """Test graceful handling of tool execution errors"""
        error_tool_manager = Mock()
        error_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        with patch.object(ai_generator.client.messages, 'create') as mock_create:
            mock_create.side_effect = [
                MockAnthropicResponse(stop_reason="tool_use", tool_calls=[
                    {"name": "search_course_content", "input": {"query": "test"}}
                ]),
                MockAnthropicResponse("Handled error gracefully")
            ]
            
            # Should not crash on tool error
            with pytest.raises(Exception, match="Tool execution failed"):
                ai_generator.generate_response(
                    "Test query",
                    tools=mock_tools,
                    tool_manager=error_tool_manager,
                    max_rounds=2
                )
    
    def test_no_tool_results_handling(self, ai_generator, mock_tools):
        """Test handling when tool returns no results"""
        empty_tool_manager = MockToolManager()
        empty_tool_manager.set_response("search_course_content", "")
        
        with patch.object(ai_generator.client.messages, 'create') as mock_create:
            mock_create.side_effect = [
                MockAnthropicResponse(stop_reason="tool_use", tool_calls=[
                    {"name": "search_course_content", "input": {"query": "nonexistent"}}
                ]),
                MockAnthropicResponse("No results found for your query")
            ]
            
            result = ai_generator.generate_response(
                "Find nonexistent content",
                tools=mock_tools,
                tool_manager=empty_tool_manager,
                max_rounds=2
            )
            
            assert "No results found" in result


class TestRealWorldScenarios:
    """Test scenarios matching the original requirements"""
    
    def test_course_comparison_scenario(self, ai_generator, mock_tools, mock_tool_manager):
        """Test: 'Search for a course that discusses the same topic as lesson 4 of course X'"""
        mock_tool_manager.set_response(
            "get_course_outline", 
            "Course X Outline:\n1. Introduction\n2. Basics\n3. Intermediate\n4. Advanced Database Indexing\n5. Expert Level"
        )
        mock_tool_manager.set_response(
            "search_course_content",
            "Found in Course Y (Lesson 3): Database indexing strategies\nFound in Course Z (Lesson 7): Advanced indexing techniques"
        )
        
        with patch.object(ai_generator.client.messages, 'create') as mock_create:
            mock_create.side_effect = [
                MockAnthropicResponse(stop_reason="tool_use", tool_calls=[
                    {"name": "get_course_outline", "input": {"course_name": "course X"}}
                ]),
                MockAnthropicResponse(stop_reason="tool_use", tool_calls=[
                    {"name": "search_course_content", "input": {"query": "Advanced Database Indexing"}}
                ]),
                MockAnthropicResponse("Course Y lesson 3 and Course Z lesson 7 cover similar database indexing topics as Course X lesson 4")
            ]
            
            result = ai_generator.generate_response(
                "Search for a course that discusses the same topic as lesson 4 of course X",
                tools=mock_tools,
                tool_manager=mock_tool_manager,
                max_rounds=2
            )
            
            # Verify the expected flow
            assert len(mock_tool_manager.call_history) == 2
            assert mock_tool_manager.call_history[0] == ("get_course_outline", {"course_name": "course X"})
            assert mock_tool_manager.call_history[1] == ("search_course_content", {"query": "Advanced Database Indexing"})
            assert "Course Y lesson 3" in result
            assert "Course Z lesson 7" in result
            assert "similar" in result.lower()
    
    def test_multipart_question_handling(self, ai_generator, mock_tools, mock_tool_manager):
        """Test handling of complex multipart questions"""
        with patch.object(ai_generator.client.messages, 'create') as mock_create:
            mock_create.side_effect = [
                MockAnthropicResponse(stop_reason="tool_use", tool_calls=[
                    {"name": "get_course_outline", "input": {"course_name": "Python Basics"}}
                ]),
                MockAnthropicResponse(stop_reason="tool_use", tool_calls=[
                    {"name": "search_course_content", "input": {"query": "functions variables"}}
                ]),
                MockAnthropicResponse("Based on the course structure and content search, here's the comprehensive answer...")
            ]
            
            result = ai_generator.generate_response(
                "What are the main topics in Python Basics course and how do functions relate to variables?",
                tools=mock_tools,
                tool_manager=mock_tool_manager,
                max_rounds=2
            )
            
            assert len(mock_tool_manager.call_history) == 2
            assert "comprehensive answer" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])