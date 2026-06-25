# Copyright 2026 — SOD benchmark (AIME-TIR) integration for SDAR. Apache-2.0.
"""Prompt templates for the math_tool (AIME tool-integrated-reasoning) environment.

The ``{task_description}`` is the *full* SOD AIME prompt (already containing the
"\\boxed{...}" final-answer instruction and the tool guidance), so these wrappers
only add the turn framing + the hermes tool-call protocol reminder. The
``_NO_HIS`` variant is used on the first turn (and when history is disabled); the
history variant prepends the running transcript of (action, tool_response) pairs.
"""

MATH_TOOL_TEMPLATE_NO_HIS = """{task_description}

You may use a Python code interpreter tool. To call it, emit a single tool call in this exact format:
<tool_call>{{"name": "code_interpreter", "arguments": {{"code": "<your python code; use print(...) to show results>"}}}}</tool_call>
The tool's output is returned to you wrapped in <tool_response> </tool_response>.
When you are confident in the final answer, stop calling tools and output it as \\boxed{{...}}."""

MATH_TOOL_TEMPLATE = """{task_description}

Prior to this step, you have already taken {step_count} step(s). Below is the interaction history: each step shows your action and the tool's <tool_response> </tool_response>. History:
{memory_context}

Now continue. To call the code interpreter, emit:
<tool_call>{{"name": "code_interpreter", "arguments": {{"code": "<your python code; use print(...) to show results>"}}}}</tool_call>
When you are confident in the final answer, stop calling tools and output it as \\boxed{{...}}."""
