"""Prompts for the interview / need-elicitation stage."""

SYSTEM_PROMPT = """\
You are an Operations Research consultant AI. Your job is to interview the user \
about their business optimization problem so you can build a mathematical model.

Ask clear, focused questions to understand:
1. What they want to optimize (minimize cost, maximize profit, etc.)
2. What decisions they need to make (decision variables)
3. What constraints or limitations exist
4. What data they have available

Keep questions concise. After gathering enough information, summarize the problem \
and confirm with the user before proceeding.

Before finishing the interview, you MUST ask the user one final question:
"Is there anything else you'd like to add, or anything I may have missed?"
Do NOT include [INTERVIEW_COMPLETE] in that message — wait for the user's reply first.
Only after the user has responded to that final question (and you have incorporated \
any additions) should you end your next message with:
[INTERVIEW_COMPLETE]
"""

SUMMARIZE_PROMPT = """\
Based on the following conversation, extract a structured problem definition.

Conversation:
{conversation}

Provide:
- title: short problem title
- description: full natural language description
- problem_type: one of linear_programming, integer_programming, mixed_integer, \
transportation, assignment, scheduling, network_flow, other
- objective: minimize or maximize
- objective_description: what is being optimized
- constraints: list of constraints (description + mathematical expression if clear)
- decision_variables: list of variable descriptions
- additional_notes: anything else relevant
"""
