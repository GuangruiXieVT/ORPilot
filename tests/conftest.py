def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "semantic: marks RAG tests that require an embedding model (skip unless --semantic)",
    )
    config.addinivalue_line(
        "markers",
        "hybrid: marks RAG tests that use full hybrid BM25+embedding retrieval (skip unless --hybrid)",
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
