def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "semantic: marks RAG tests that require an embedding model (skip unless --semantic)",
    )
    config.addinivalue_line(
        "markers",
        "hybrid: marks RAG tests that use full hybrid BM25+embedding retrieval (skip unless --hybrid)",
    )
    config.addinivalue_line(
        "markers",
        "llm: marks tests that call a real LLM (skip unless --llm; needs api_key in orpilot.toml)",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--semantic",
        action="store_true",
        default=False,
        help="Run RAG tests that require an embedding model (needs ANTHROPIC_API_KEY)",
    )
    parser.addoption(
        "--hybrid",
        action="store_true",
        default=False,
        help="Run hybrid BM25+embedding retrieval tests (needs embed_api_key in orpilot.toml or env)",
    )
    parser.addoption(
        "--llm",
        action="store_true",
        default=False,
        help="Run tests that call a real LLM (needs api_key in orpilot.toml)",
    )
