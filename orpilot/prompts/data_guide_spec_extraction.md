---
version: 1.0.0
---

Extract the CSV file specifications from the conversation below. The agent has defined which CSV files the user needs to provide.

Conversation:
{conversation}

Return a JSON list of file specs. Each spec should have:
- filename: the CSV filename
- description: what the file contains
- columns: list of {{name, dtype, description}}
- optional: true if the agent described the file as optional (e.g. "If omitted, defaults to 0"), false otherwise
