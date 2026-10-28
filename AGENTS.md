# Guidance for AGENTS

## About the Repo

This repository is a simplified version of ChatDKU, a university AI assistant that helps students and staff find answers within long documents.

It is intended for applicants who are interested in joining the team and taking initiative in further developing the product.

The purpose of the exercise is not only to produce working code, but also to demonstrate clear engineering judgment, understanding of the existing codebase, and the ability to explain technical decisions.

## Requirements for the Agent

- Do not make substantial or unnecessary changes to the repository.
- Prefer small, focused changes that preserve the existing architecture and coding style.
- Do not commit changes on behalf of the user.
- Do not add, persist, or commit data ingested or uploaded by the user.
- Do not replace existing implementations simply because another approach appears cleaner. Understand the current implementation first.

## Development Process

Before making a non-trivial code change:

1. Inspect the relevant parts of the codebase.
2. Explain what the existing code does and how the relevant components interact.
3. State the problem you intend to solve.
4. Describe the proposed change and why it is appropriate for the existing architecture.
5. Identify any important assumptions, tradeoffs, or possible side effects.

While implementing:

- Make changes incrementally rather than rewriting large sections at once.
- Explain important pieces of code as they are introduced.
- For non-obvious logic, explain why the implementation works rather than only describing what the syntax does.
- Reuse existing abstractions and utilities where reasonable.
- Avoid introducing dependencies unless they are clearly justified.

After implementing:

- Summarize exactly what changed.
- Explain the resulting execution or data flow.
- Describe how the change was tested or verified.
- Mention known limitations, edge cases, or follow-up improvements.
- If tests were not run or something could not be verified, say so explicitly.

## Code Explanation Expectations

The agent should help the user understand the code, not merely generate it.

For meaningful changes, explanations should cover:

- the role of each affected component;
- how data moves through the relevant code;
- why the chosen implementation was used;
- how it integrates with the rest of the system;
- important failure cases or edge cases; and
- alternatives considered when there is a meaningful design tradeoff.

Avoid unexplained “magic” changes. A working implementation without a clear explanation of how and why it works is incomplete.

## Scope and Initiative

The agent may point out architectural issues or potential improvements, but should not implement unrelated refactors without being asked.

When a larger redesign appears useful, explain the proposed redesign separately instead of silently expanding the scope of the task.