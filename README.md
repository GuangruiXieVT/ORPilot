# AIFOR — AI Operations Research Agent

LLM-powered Operations Research modeling and solving.

## Installation

```bash
pip install -e ".[all-solvers]"
```

## Quick Start

```python
from aifor import Agent

agent = Agent(llm_provider="openai", solver="pulp")
result = agent.run()
```

## CLI

```bash
aifor run --provider openai --solver pulp
aifor config
```
