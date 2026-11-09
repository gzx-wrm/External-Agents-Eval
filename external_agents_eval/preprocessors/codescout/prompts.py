"""CodeScout Appendix A prompt templates (line wrapping/bullets normalized).

Source: Suri et al., CodeScout: Contextual Problem Statement Enhancement for
Software Agents, https://arxiv.org/abs/2603.05744v2, Appendix A.
The arXiv manuscript is licensed CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/
Authors: Manan Suri, Xiangci Li, Mehdi Shojaie, Songyang Han, Chao-Chun Hsu,
Shweta Garg, Aniket Anand Deshmukh, and Varun Kumar.
The prompt text is reproduced; repository representation/runtime are our implementation.
"""

SCOPE = """You are analyzing a software bug report to identify
which files and code components to investigate.

PROBLEM STATEMENT:
{problem_statement}

REPOSITORY STRUCTURE:
{file_tree}

Task: Identify the most promising files, classes, and
functions to explore for understanding this issue.
Focus on:
- Files/classes mentioned in the problem statement
- Related components that might be affected
- Common entry points and core functionality
- Test files that might reveal expected behavior

Return exactly 5–10 targets in this format:
1. target_type:file,
target_name:exact_name,
reasoning:why_relevant
2. target_type:class,
target_name:exact_name,
reasoning:why_relevant
3. target_type:function,
target_name:exact_name,
reasoning:why_relevant

Be specific with names—use exact file paths and
class/function names from the tree. Each entry must
be on a separate line and follow the exact format
shown."""

ANALYZE = """You are analyzing code content to understand how
it relates to a specific bug report. Your analysis will
directly feed into an augmented problem statement.

ORIGINAL PROBLEM:
{problem_statement}

CODE CONTENT:
{content}

TARGET INFO: {target_type} - {target_name}

Provide analysis in the following format:
RELEVANCE: X/10
ROLE_IN_ISSUE:
Describe how this code relates to the reported issue.
What is its role in causing or manifesting the bug?
FIX_LOCATION_HINTS:
Identify specific locations where fixes might be
needed (line numbers, methods, conditions). What
type of changes might be required (validation, error
handling, logic fixes)?
EXPLORATION_INSIGHTS:
Surface key technical details that help understand
the issue—algorithms, data flow, dependencies, code
patterns, or architectural decisions relevant to the bug.
Note test scenarios or edge cases this code suggests
should be examined.
ALTERNATIVE_HYPOTHESES:
Consider alternative explanations for the bug based
on this code. What related components might also
be involved? What different root causes does this
analysis suggest?

Note: Focus on insights that will help create actionable
exploration hints and fix suggestions. If relevance < 4,
keep all sections brief but still provide analysis."""

SYNTHESIZE = """You are creating an enhanced problem statement that
will help developers quickly understand and fix a
software issue.

ORIGINAL PROBLEM STATEMENT:
{original_ps}

CODE ANALYSIS RESULTS: For each analyzed target:
{insights}

Instructions: Synthesize these analyses into a comprehensive
problem statement. Include all relevant
details from the original statement. Use the code
analysis as context to inform your narrative—don't
simply copy it verbatim.

Structure your output with these sections:
## Issue Description: Rewrite the core problem
clearly, incorporating technical insights.
## Reproduction Steps: Include original steps if
available. If the original issue includes code, preserve
it here—this is essential. Add technical details about
what happens internally. Include relevant stack traces
or error patterns. Suggest additional reproduction
scenarios if helpful.
## Expected Behavior: Describe what should happen
instead, informed by code analysis.
## Exploration Hints:
- Files to Examine: file1.py - why relevant (role
in issue, key findings)
- Key Classes/Functions: ClassName.method() -
what to look for
- Areas of Interest: Specific code patterns or logic to
investigate
## Fix Hints:
- High Confidence Locations: Where fixes are likely
needed and why
- Implementation Suggestions: Concrete fix ideas
with limitations
- Alternative Hypotheses: Other potential root
causes

Keep technical language precise but accessible. Focus
on actionable insights."""
