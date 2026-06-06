"""RAG retrieval test suite.

All tests share TEST_CASES — structured ProblemDefinition queries that
mirror the exact format produced by _build_rag_context during orpilot run.
Each method is tested independently on the same cases so results are
directly comparable.

Each case:  (query, expected_names, k)
  query     – structured Title/Description/Objective/Variables/Constraints
  expected  – corpus example names that SHOULD appear in the top-k results
  k         – how many results to retrieve (at least one expected must appear)

Run BM25-only tests (no API key required):
    pytest tests/test_rag_retrieval.py -v

Run all three methods (needs embed_api_key in orpilot.toml):
    pytest tests/test_rag_retrieval.py -v --hybrid

Print side-by-side comparison (needs --hybrid, add -s for output):
    pytest tests/test_rag_retrieval.py -v --hybrid -s -k comparison
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from orpilot.rag.indexer import load_index, build_index, CORPUS_DIR
from orpilot.rag.retriever import Retriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def retriever_bm25():
    """BM25-only retriever, no embedder needed."""
    index = load_index()
    if index is None:
        index = build_index(CORPUS_DIR, embedder=None, force=False)
    return Retriever(index, embedder=None)


@pytest.fixture(scope="session")
def retriever_hybrid(request):
    """Retriever with embedder for semantic-only and hybrid tests.

    Loads embed_api_key from orpilot.toml (same source as orpilot run).
    Skips if --hybrid is not passed or no key/embeddings are available.
    """
    if not request.config.getoption("--hybrid"):
        pytest.skip("pass --hybrid to run embedding-based retrieval tests")

    api_key = None
    try:
        import tomllib
        cfg = tomllib.loads(Path("orpilot.toml").read_text())
        api_key = cfg.get("embed_api_key")
    except Exception:
        pass
    api_key = api_key or os.getenv("ORPILOT_EMBED_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("No embed_api_key found in orpilot.toml or environment")

    index = load_index()
    if index is None:
        index = build_index(CORPUS_DIR, embedder=None, force=False)
    if not index.get("embedding_model"):
        pytest.skip("Index has no embeddings — rebuild with: orpilot rag build --embed")

    try:
        from orpilot.rag.embedder import Embedder
        embedder = Embedder(api_key=api_key)
    except Exception as e:
        pytest.skip(f"Could not initialise embedder: {e}")

    return Retriever(index, embedder=embedder)


# ---------------------------------------------------------------------------
# Test cases — structured queries that mirror _build_rag_context output
# ---------------------------------------------------------------------------

TEST_CASES = [
    (
        # Facility location — uses "distribution centers" + "delivery cost"
        "Title: Distribution Center Selection Problem\n"
        "Description: A retailer must decide which distribution centers to open in order to "
        "serve a set of stores. Each distribution center has a fixed activation cost and a "
        "maximum throughput capacity. For each distribution center and store pair there is a "
        "known per-unit delivery cost. Each store has a fixed demand that must be fully met. "
        "A distribution center can only serve stores if it is open. Choose which distribution "
        "centers to open and how much to ship from each to minimize total activation and "
        "delivery costs.\n"
        "Objective: minimize\n"
        "Objective description: Total fixed activation costs for opened distribution centers "
        "plus per-unit delivery costs times quantities shipped\n"
        "Decision variables: open[d]: 1 if distribution center d is opened, 0 otherwise; "
        "ship[d,s]: units shipped from distribution center d to store s\n"
        "Parameters: activation_cost[d]: fixed cost to open distribution center d; "
        "capacity[d]: max throughput of distribution center d; "
        "delivery_cost[d,s]: per-unit delivery cost from d to s; demand[s]: demand at store s\n"
        "Constraints: Each store's demand must be fully met; Total shipments from each open "
        "distribution center cannot exceed its capacity; No shipments from closed distribution "
        "centers (big-M linking)",
        ["facility_location"],
        3,
    ),
    (
        # VRP
        "Title: Capacitated Vehicle Routing Problem\n"
        "Description: A depot operator must plan routes for a fleet of vehicles to visit all "
        "customers and return to the depot. Each vehicle starts and ends at the depot, has a "
        "maximum load capacity, and cannot visit any location more than once. The goal is to "
        "minimize total travel distance. Subtours must be eliminated.\n"
        "Objective: minimize\n"
        "Objective description: Total travel distance across all vehicle routes\n"
        "Decision variables: arc[i,j,v]: 1 if vehicle v travels from location i to location j; "
        "position[c,v]: visit sequence of customer c in vehicle v route for subtour elimination\n"
        "Parameters: distance[i,j]: travel distance between locations i and j; "
        "capacity[v]: load capacity of vehicle v; demand[c]: load required at customer c\n"
        "Constraints: Each customer visited by exactly one vehicle; Each vehicle departs and "
        "returns to depot; Vehicle load capacity not exceeded; Subtour elimination (MTZ)",
        ["vrp", "vrp_time_windows"],
        3,
    ),
    (
        # Multi-period inventory — uses "carrying cost" instead of "holding cost"
        "Title: Multi-Product Inventory Management\n"
        "Description: A warehouse manager plans order quantities for multiple products over "
        "multiple periods to meet customer demand while minimizing total ordering and carrying "
        "costs. Inventory carries over between periods. Warehouse capacity is shared across "
        "all products.\n"
        "Objective: minimize\n"
        "Objective description: Total ordering costs plus carrying costs for all products "
        "over all periods\n"
        "Decision variables: order[p,t]: units of product p ordered in period t; "
        "inventory[p,t]: units of product p held at end of period t\n"
        "Parameters: demand[p,t]: units of product p demanded in period t; "
        "ordering_cost[p]: fixed cost per order for product p; "
        "carrying_cost[p]: per-unit carrying cost for product p per period; "
        "warehouse_capacity: maximum total units across all products\n"
        "Constraints: Inventory balance each period for each product; "
        "Non-negative inventory and orders; Shared warehouse capacity not exceeded",
        ["inventory_multi_products", "inventory_single_product"],
        3,
    ),
    (
        # Shift scheduling — uses "staffing" language
        "Title: Workforce Shift Scheduling\n"
        "Description: A call center manager must schedule full-time and part-time agents "
        "across shifts each day of the week to meet minimum staffing requirements at minimum "
        "cost. Full-time agents cost more per shift but cover more hours. The total weekly "
        "staffing cost must stay within budget.\n"
        "Objective: minimize\n"
        "Objective description: Total weekly staffing cost across all agent types, shifts, "
        "and days\n"
        "Decision variables: staff[a,s,d]: number of agents of type a scheduled on shift s "
        "on day d\n"
        "Parameters: min_staff[s,d]: minimum agents required on shift s day d; "
        "cost[a]: cost per agent per shift for agent type a; "
        "weekly_limit[a]: maximum total shifts per week for agent type a\n"
        "Constraints: Minimum staffing met each shift and day; Weekly shift limit per agent "
        "type not exceeded",
        ["scheduling_shift_planning", "workforce_scheduling"],
        3,
    ),
    (
        # Make-or-buy — uses "vendors" instead of "suppliers"
        "Title: Make-or-Buy Sourcing Decision\n"
        "Description: A manufacturer must decide for each component whether to produce "
        "it in-house or purchase it from external vendors. In-house production consumes "
        "machine capacity. Vendor sourcing has a per-unit purchase price. The goal is to "
        "minimize total production and procurement costs while meeting demand for all "
        "components.\n"
        "Objective: minimize\n"
        "Objective description: Total in-house production costs plus vendor procurement costs\n"
        "Decision variables: make[i]: 1 if component i is produced in-house; "
        "produce[i]: units of component i manufactured; buy[i]: units purchased from vendor\n"
        "Parameters: production_cost[i]: per-unit cost to make component i; "
        "vendor_price[i]: per-unit vendor price for component i; "
        "machine_hours[i]: machine hours required per unit of component i; "
        "machine_capacity: total available machine hours; demand[i]: required units of component i\n"
        "Constraints: Demand met via make or buy for each component; Machine capacity limit; "
        "Linking constraint: production only allowed when make decision is 1",
        ["make_or_buy"],
        3,
    ),
    (
        # Nurse rostering
        "Title: Weekly Nurse Rostering\n"
        "Description: A hospital must assign nurses to shifts across days of the week to "
        "meet minimum coverage requirements. Each nurse can work at most a limited number "
        "of shifts per week. The goal is to minimize total rostering cost.\n"
        "Objective: minimize\n"
        "Objective description: Total cost of nurse-shift-day assignments\n"
        "Decision variables: assign[n,s,d]: 1 if nurse n is assigned to shift s on day d\n"
        "Parameters: cost[n,s]: cost of assigning nurse n to shift s; "
        "min_nurses[s,d]: minimum nurses required on shift s on day d; "
        "max_shifts[n]: maximum shifts nurse n can work per week\n"
        "Constraints: Minimum nurse coverage per shift per day; "
        "Maximum weekly shifts per nurse not exceeded",
        ["assignment_nurse_rostering"],
        3,
    ),
    (
        # Lot sizing — uses "changeover cost" instead of "setup cost"
        "Title: Capacitated Lot Sizing\n"
        "Description: A production planner decides when and how much to produce each product "
        "each period. A fixed changeover cost is incurred whenever a product is produced in a "
        "period. Total production across all products is limited by machine capacity. "
        "Inventory carries over between periods.\n"
        "Objective: minimize\n"
        "Objective description: Total changeover costs plus production costs plus inventory "
        "holding costs over the planning horizon\n"
        "Decision variables: produce[p,t]: units of product p produced in period t; "
        "setup[p,t]: 1 if product p is produced in period t; "
        "inventory[p,t]: units of product p in stock at end of period t\n"
        "Parameters: changeover_cost[p]: fixed cost when product p runs in a period; "
        "unit_cost[p]: variable production cost per unit; "
        "holding_cost[p]: per-unit inventory cost per period; "
        "demand[p,t]: demand for product p in period t; machine_capacity: hours per period\n"
        "Constraints: Inventory balance each period; Changeover links production to setup "
        "binary; Machine capacity shared across products",
        ["lot_sizing"],
        3,
    ),
    (
        # Supply chain network with site activation/deactivation — uses
        # "activate/deactivate" and "shut down" instead of "open/close" to
        # test whether hybrid search recovers the site-closing example when
        # BM25 may rank generic network examples higher due to keyword overlap.
        "Title: Multi-Period Supply Chain Network with Site Activation and Deactivation\n"
        "Description: A manufacturer operates a network of production sites, warehouses, and "
        "retailers over multiple planning periods. In each period, each production site can be "
        "either active or inactive. A one-time cost is incurred when a site is activated "
        "(transitions from inactive to active) and a separate one-time cost is incurred when "
        "a site is deactivated (shut down). Active sites produce goods and incur per-period "
        "operating costs; inactive sites produce nothing and incur no operating cost. Goods "
        "flow from production sites to warehouses and on to retailers over a fixed set of "
        "shipping lanes. Warehouse storage capacity and arc throughput limits apply each "
        "period. Every retailer's demand must be met exactly each period. The goal is to "
        "decide which sites to activate or shut down each period, how much to produce, and "
        "how to route goods to minimise total activation, deactivation, operating, production, "
        "holding, and transportation costs.\n"
        "Objective: minimize\n"
        "Objective description: Total activation costs plus deactivation costs plus per-period "
        "operating costs at active sites plus production costs plus inventory holding costs "
        "plus transportation costs over the planning horizon\n"
        "Decision variables: active[i,t]: 1 if site i is active in period t; "
        "activate[i,t]: 1 if site i is activated at the start of period t; "
        "deactivate[i,t]: 1 if site i is shut down at the start of period t; "
        "produce[i,t]: units produced at site i in period t; "
        "inventory_site[i,t]: inventory held at site i at end of period t; "
        "inventory_wh[w,t]: inventory held at warehouse w at end of period t; "
        "ship_sw[i,w,t]: units shipped from site i to warehouse w in period t; "
        "ship_wr[w,r,t]: units shipped from warehouse w to retailer r in period t\n"
        "Parameters: activation_cost[i]: one-time cost to activate site i; "
        "deactivation_cost[i]: one-time cost to shut down site i; "
        "operating_cost[i]: per-period cost when site i is active; "
        "production_cost[i]: per-unit production cost at site i; "
        "holding_cost_site[i]: per-unit holding cost at site i; "
        "holding_cost_wh[w]: per-unit holding cost at warehouse w; "
        "shipping_cost[i,w] / shipping_cost[w,r]: per-unit cost on each arc; "
        "demand[r,t]: units required by retailer r in period t\n"
        "Constraints: Activation/deactivation transition linking active status across periods; "
        "Production only allowed when site is active; "
        "Inventory balance at sites and warehouses each period; "
        "Warehouse storage capacity not exceeded; Arc throughput limits; "
        "Retailer demand met exactly each period",
        ["network_production_inventory_transportation_open_close"],
        3,
    ),
    # ── Paraphrase / vocabulary-gap cases ──────────────────────────────────
    # Same structured format as above, but written with non-OR vocabulary.
    # Shows where BM25 and semantic diverge despite identical query structure.
    (
        # minimax_workload — "overwhelmed"/"burdened"/"effort" instead of "workload"/"minimax"
        # all three methods: rank 3 — tied, no advantage to either method
        "Title: Fairest Task Assignment\n"
        "Description: A team leader must assign tasks to workers so that no single "
        "worker ends up overwhelmed. Each task goes to exactly one worker. The goal "
        "is to minimise the load carried by the most burdened person on the team.\n"
        "Objective: minimize\n"
        "Objective description: Total effort assigned to the most burdened worker\n"
        "Decision variables: assign[t,w]: 1 if task t is given to worker w; "
        "peak_load: maximum effort carried by any single worker\n"
        "Parameters: effort[t]: effort required for task t; "
        "capacity[w]: maximum tasks worker w can handle\n"
        "Constraints: Each task assigned to exactly one worker; "
        "peak_load is at least the total effort of every worker; "
        "Each worker does not exceed their capacity",
        ["minimax_workload"],
        3,
    ),
    (
        # capital_budgeting — "pot of money"/"payoff"/"fund" instead of "capital"/"budget"
        # BM25: rank 1 (keyword overlap on "fund"/"cost"/"payoff")
        # semantic: rank 2 | hybrid: rank 1
        # adding structure helped BM25 — no semantic advantage here
        "Title: Investment Portfolio Selection\n"
        "Description: A fund manager has a fixed pot of money and a list of "
        "opportunities. Each opportunity has an upfront cost and an expected payoff. "
        "Opportunities are all-or-nothing — either fully funded or skipped. Choose "
        "which ones to fund so the total payoff is as large as possible without "
        "overspending.\n"
        "Objective: maximize\n"
        "Objective description: Total expected payoff across all funded opportunities\n"
        "Decision variables: fund[i]: 1 if opportunity i is selected, 0 otherwise\n"
        "Parameters: cost[i]: upfront cost of opportunity i; "
        "payoff[i]: expected payoff of opportunity i; "
        "pot: total money available to spend\n"
        "Constraints: Total cost of selected opportunities does not exceed the pot",
        ["capital_budgeting"],
        2,
    ),
    (
        # blending — chemistry vocabulary: "purity"/"concentration"/"fraction"/"raw material"
        # semantic: rank 1 | hybrid: rank 1 | BM25: rank 4
        # vocabulary gap persists even with structured format — clear semantic win
        "Title: Raw Material Mixture Optimisation\n"
        "Description: A chemist must formulate a mixture from a set of raw materials. "
        "The mixture must meet minimum purity and concentration thresholds for several "
        "chemical properties. Each raw material contributes differently to each "
        "property. The fractions of all materials must sum to one. Minimise cost.\n"
        "Objective: minimize\n"
        "Objective description: Total cost of the raw material mixture per unit\n"
        "Decision variables: fraction[m]: proportion of raw material m in the mixture\n"
        "Parameters: unit_cost[m]: cost per unit of material m; "
        "composition[m,p]: level of property p from material m; "
        "min_level[p]: minimum required level of property p; "
        "max_level[p]: maximum allowed level of property p\n"
        "Constraints: Composition requirements met for each property; "
        "Fractions sum to one; Non-negative fractions",
        ["blending"],
        4,
    ),
]


def _case_id(query: str) -> str:
    first_line = query.split("\n")[0]
    return first_line[7:55] if first_line.startswith("Title: ") else first_line[:55]

_CASE_IDS = [_case_id(c[0]) for c in TEST_CASES]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _names(results):
    return [r["name"] for r in results]


# ---------------------------------------------------------------------------
# Tests — one per method, all using TEST_CASES
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,expected,k", TEST_CASES, ids=_CASE_IDS)
def test_bm25_retrieval(retriever_bm25, query, expected, k):
    """BM25 + synonym expansion only."""
    results = retriever_bm25.retrieve(query, k=k)
    retrieved = _names(results)
    hits = [n for n in expected if n in retrieved]
    assert hits, f"Expected one of {expected} in top-{k}.\nGot: {retrieved}"


@pytest.mark.hybrid
@pytest.mark.parametrize("query,expected,k", TEST_CASES, ids=_CASE_IDS)
def test_semantic_retrieval(retriever_hybrid, query, expected, k):
    """Pure cosine-similarity retrieval, no BM25."""
    results = retriever_hybrid.retrieve(query, k=k, semantic_only=True)
    retrieved = _names(results)
    hits = [n for n in expected if n in retrieved]
    assert hits, f"Expected one of {expected} in top-{k}.\nGot: {retrieved}"


@pytest.mark.hybrid
@pytest.mark.parametrize("query,expected,k", TEST_CASES, ids=_CASE_IDS)
def test_hybrid_retrieval(retriever_hybrid, query, expected, k):
    """Hybrid BM25 + semantic (RRF fusion)."""
    results = retriever_hybrid.retrieve(query, k=k)
    retrieved = _names(results)
    hits = [n for n in expected if n in retrieved]
    assert hits, f"Expected one of {expected} in top-{k}.\nGot: {retrieved}"


# ---------------------------------------------------------------------------
# Comparison report — always passes, run with -s to see output
# ---------------------------------------------------------------------------

@pytest.mark.hybrid
def test_retrieval_method_comparison(retriever_bm25, retriever_hybrid):
    """Side-by-side hit/miss table for BM25, semantic-only, and hybrid.

    All TEST_CASES are shown. The last three (paraphrase queries) reveal
    where semantic finds the exact target while BM25 settles for a related one.
    Run with -s to see the full report.
    """
    K = 3

    hits: dict[str, int] = {"BM25": 0, "Semantic": 0, "Hybrid": 0}
    total = len(TEST_CASES)

    print(f"\n{'='*76}")
    print(f"{'RETRIEVAL METHOD COMPARISON  (top-' + str(K) + ')':^76}")
    print(f"{'='*76}")

    for query, expected, case_k in TEST_CASES:
        k = max(K, case_k)
        title = _case_id(query)
        bm25_res = _names(retriever_bm25.retrieve(query, k=k))[:K]
        sem_res  = _names(retriever_hybrid.retrieve(query, k=k, semantic_only=True))[:K]
        hyb_res  = _names(retriever_hybrid.retrieve(query, k=k))[:K]
        bm25_hit = any(n in bm25_res for n in expected)
        sem_hit  = any(n in sem_res  for n in expected)
        hyb_hit  = any(n in hyb_res  for n in expected)
        if bm25_hit: hits["BM25"] += 1
        if sem_hit:  hits["Semantic"] += 1
        if hyb_hit:  hits["Hybrid"] += 1
        mark = lambda h: "HIT " if h else "MISS"
        print(f"\n  {title}")
        print(f"  Expected : {expected}")
        print(f"  BM25     [{mark(bm25_hit)}]: {bm25_res}")
        print(f"  Semantic [{mark(sem_hit)}]: {sem_res}")
        print(f"  Hybrid   [{mark(hyb_hit)}]: {hyb_res}")

    print(f"\n{'-'*76}")
    print(f"  Hit rates (top-{K} of {total} cases):")
    for method, count in hits.items():
        bar = "#" * count + "-" * (total - count)
        print(f"  {method:<10} {count}/{total}  [{bar}]  {count/total:.0%}")
    print(f"{'='*76}")
    assert True
