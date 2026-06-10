"""RAG retrieval test suite.

All tests share TEST_CASES — structured ProblemDefinition queries that
mirror the exact format produced by _build_rag_context during orpilot run.
Each method is tested independently on the same cases so results are
directly comparable.

Each case:  (query, expected_names, k, patterns)
  query     – structured Title/Description/Objective/Variables/Constraints
  expected  – corpus example names that SHOULD appear in the top-k results
  k         – how many results to retrieve (at least one expected must appear)
  patterns  – VOCAB pattern names a well-functioning LLM classifier should return
               for this problem (no "pat:" prefix); used as the query fingerprint

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
# Vocabulary-poor query strings: describe mathematical structure with neutral
# language so BM25 misses but the structural Jaccard / semantic channel recovers.
# Defined before TEST_CASES because they are referenced inside it.
# ---------------------------------------------------------------------------

# Fires: minimax objective ("equalize load"), binary variable ("0-1"), scalar variable
# ("peak-load variable").  Targets: parking_minimax, inheritance_partition.
_Q1_MINIMAX = (
    "Title: Load Equalization Across Two Zones\n"
    "Description: A controller assigns items to one of two zones. Each item has a known size. "
    "For each item, a 0-1 assignment flag indicates which zone it goes to. "
    "Every item must go to exactly one zone. "
    "A peak-load variable records the larger of the two zone totals. "
    "The goal is to equalize load by minimising this peak-load variable.\n"
    "Objective: minimize\n"
    "Objective description: Peak-load variable representing the larger zone total\n"
    "Decision variables: assign[i]: 0-1, whether item i goes to zone B; "
    "peak_load: peak-load variable bounding the larger zone total\n"
    "Constraints: Peak-load variable is at least the total for each zone; "
    "Every item assigned to exactly one zone"
)

# Fires: two-stage stochastic ("here-and-now" + "recourse"), binary variable ("0-1"),
# scenario weighted sum ("expected profit").  Targets: stochastic_facility_location,
# stochastic_inventory.
_Q2_STOCHASTIC = (
    "Title: Here-and-Now 0-1 Commitment Under Uncertainty\n"
    "Description: A manager must make here-and-now 0-1 resource commitments before future "
    "outcomes are known. For each outcome a recourse adjustment compensates any shortfall. "
    "The objective minimises fixed commitment cost plus the weighted-average recourse cost, "
    "where each outcome contributes to the expected profit of avoiding shortfalls.\n"
    "Objective: minimize\n"
    "Objective description: Fixed commitment cost plus weighted recourse cost over outcomes\n"
    "Decision variables: commit[r]: 0-1, whether resource r is committed here-and-now; "
    "adjust[r,w]: continuous recourse adjustment for outcome w\n"
    "Constraints: Here-and-now commitments are locked before outcomes are revealed; "
    "Recourse variables cover any shortfall in each outcome"
)

# Fires: deviation variables ("shortage"), temporal lag ("carryover" + "stock balance"),
# init constraint ("initial stock at time zero"), temporal lag ("stock balance"), binary variable,
# big-M linking.  Targets: lot_sizing_backlog, inventory_backlogging.
_Q3_BACKLOG = (
    "Title: Multi-Step Stock Allocation with Unmet Requirements\n"
    "Description: A manager allocates a limited resource across time steps. A binary "
    "activation indicator gates each time step via a big-M constraint. If allocated "
    "resources fall short of requirements, the shortage carryover is penalised. "
    "Stock balance must hold each time step: ending stock equals previous stock plus "
    "allocation minus requirements plus shortage carryover. Initial stock at time zero "
    "is fixed. Activation incurs a fixed cost.\n"
    "Objective: minimize\n"
    "Objective description: Total allocation cost plus activation cost plus shortage "
    "carryover penalty\n"
    "Decision variables: allocate[t]: units allocated in time step t; "
    "active[t]: binary, activation indicator; "
    "stock[t]: units held at end of time step t; "
    "carryover[t]: unmet requirements carried over\n"
    "Constraints: Stock balance each time step; Initial stock at time zero fixed; "
    "Allocation only when active (big-M); Shortage carryover non-negative"
)


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
        ["binary variable", "big-M linking"],
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
        ["binary variable", "MTZ subtour elimination", "duplicate set", "exclude diagonal"],
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
        ["temporal lag", "balance constraint"],
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
        ["scheduling_shift_planning", "workforce_scheduling",
         "cyclic_staff_rostering", "store_shift", "restaurant_shift"],
        3,
        ["integer variable"],
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
        ["binary variable", "big-M linking"],
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
        ["binary variable"],
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
        ["binary variable", "temporal lag", "big-M linking"],
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
        ["binary variable", "temporal lag", "big-M linking", "implication constraint"],
    ),
    # ── Paraphrase / vocabulary-gap cases ──────────────────────────────────
    # Same structured format as above, but written with non-OR vocabulary.
    # Shows where BM25 and semantic diverge despite identical query structure.
    (
        # minimax_workload — "overwhelmed"/"burdened"/"effort" instead of "workload"/"minimax"
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
        ["binary variable", "minimax objective", "scalar variable"],
    ),
    (
        # capital_budgeting — "pot of money"/"payoff"/"fund" instead of "capital"/"budget"
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
        ["binary variable"],
    ),
    (
        # blending — chemistry vocabulary: "purity"/"concentration"/"fraction"/"raw material"
        # semantic: rank 1 | hybrid: rank 1 | BM25: struggles (vocabulary gap)
        # No structural patterns unique to blending — retrieval relies on semantic channel.
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
        ["blending", "sulfur_blending"],
        5,
        ["balance constraint"],
    ),
    # ── Vocabulary-poor queries: BM25 misses, structural Jaccard / semantic recovers ──
    # These queries describe mathematical structure with neutral language, avoiding all
    # domain vocabulary from the target corpus examples.  BM25 is expected to miss;
    # the structural Jaccard channel (and/or embeddings) provides the recovery signal.
    (
        _Q1_MINIMAX,
        ["parking_minimax", "inheritance_partition", "minimax_workload"],
        3,
        ["binary variable", "minimax objective", "scalar variable"],
    ),
    (
        _Q2_STOCHASTIC,
        ["stochastic_facility_location", "stochastic_inventory"],
        3,
        ["binary variable", "two-stage stochastic", "scenario weighted sum",
         "deviation variables"],
    ),
    (
        _Q3_BACKLOG,
        ["lot_sizing_backlog", "inventory_backlogging"],
        3,
        ["binary variable", "temporal lag", "init constraint",
         "big-M linking", "deviation variables", "balance constraint"],
    ),
]


def _case_id(query: str) -> str:
    first_line = query.split("\n")[0]
    return first_line[7:55] if first_line.startswith("Title: ") else first_line[:55]

_CASE_IDS = [_case_id(c[0]) for c in TEST_CASES]

# IDs of cases where BM25 alone is expected to miss (vocabulary-poor by design).
_BM25_MISS_IDS = {_case_id(q) for q in (_Q1_MINIMAX, _Q2_STOCHASTIC, _Q3_BACKLOG)}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _names(results):
    return [r["name"] for r in results]


def _fp(patterns: list[str]) -> set[str]:
    """Build modeling pattern tokens from VOCAB pattern names (no 'pat:' prefix)."""
    return {f"pat:{p}" for p in patterns}


# ---------------------------------------------------------------------------
# Tests — one per method, all using TEST_CASES
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("query,expected,k,patterns", TEST_CASES, ids=_CASE_IDS)
def test_bm25_retrieval(retriever_bm25, query, expected, k, patterns, request):
    """BM25 + synonym expansion only.

    Vocabulary-poor cases (in _BM25_MISS_IDS) are xfail: BM25 misses them by
    design; they need the Jaccard or semantic channel to be recovered.
    """
    if request.node.callspec.id in _BM25_MISS_IDS:
        pytest.xfail("vocabulary-poor query: BM25 misses by design; needs Jaccard or semantic channel")
    results = retriever_bm25.retrieve(query, k=k)
    retrieved = _names(results)
    hits = [n for n in expected if n in retrieved]
    assert hits, f"Expected one of {expected} in top-{k}.\nGot: {retrieved}"


@pytest.mark.hybrid
@pytest.mark.parametrize("query,expected,k,patterns", TEST_CASES, ids=_CASE_IDS)
def test_semantic_retrieval(retriever_hybrid, query, expected, k, patterns):
    """Pure cosine-similarity retrieval, no BM25."""
    results = retriever_hybrid.retrieve(query, k=k, semantic_only=True)
    retrieved = _names(results)
    hits = [n for n in expected if n in retrieved]
    assert hits, f"Expected one of {expected} in top-{k}.\nGot: {retrieved}"


@pytest.mark.hybrid
@pytest.mark.parametrize("query,expected,k,patterns", TEST_CASES, ids=_CASE_IDS)
def test_hybrid_retrieval(retriever_hybrid, query, expected, k, patterns):
    """Hybrid BM25 + semantic (RRF fusion)."""
    results = retriever_hybrid.retrieve(query, k=k)
    retrieved = _names(results)
    hits = [n for n in expected if n in retrieved]
    assert hits, f"Expected one of {expected} in top-{k}.\nGot: {retrieved}"


@pytest.mark.parametrize("query,expected,k,patterns", TEST_CASES, ids=_CASE_IDS)
def test_structural_retrieval(retriever_bm25, query, expected, k, patterns):
    """BM25 + structural Jaccard fingerprint (no embedder needed).

    Uses k+2 to allow for the small rank shifts the Jaccard channel can
    introduce when common patterns (binary variable, coverage) appear in
    many corpus examples and dilute the BM25 signal.
    """
    results = retriever_bm25.retrieve(query, k=k + 2, modeling_patterns=_fp(patterns))
    retrieved = _names(results)
    hits = [n for n in expected if n in retrieved]
    assert hits, f"Expected one of {expected} in top-{k+2}.\nGot: {retrieved}"


@pytest.mark.hybrid
@pytest.mark.parametrize("query,expected,k,patterns", TEST_CASES, ids=_CASE_IDS)
def test_full_hybrid_retrieval(retriever_hybrid, query, expected, k, patterns):
    """Full three-channel retrieval: BM25 + semantic + structural Jaccard (RRF fusion).

    This is the actual production path used by ir_builder_node.
    """
    results = retriever_hybrid.retrieve(query, k=k, modeling_patterns=_fp(patterns))
    retrieved = _names(results)
    hits = [n for n in expected if n in retrieved]
    assert hits, f"Expected one of {expected} in top-{k}.\nGot: {retrieved}"


# ---------------------------------------------------------------------------
# Comparison reports — always pass; run with -s to see output
# ---------------------------------------------------------------------------

def test_bm25_vs_structural_comparison(retriever_bm25):
    """Non-hybrid comparison: BM25 alone vs BM25+Jaccard for all TEST_CASES.

    Shows for each case which patterns were detected and whether adding the
    Jaccard channel changed the result. Runs without --hybrid (no API key).
    Run with -s to see the full report.
    """
    K = 3
    hits = {"BM25": 0, "BM25+Jaccard": 0}
    jaccard_activated = 0
    jaccard_lifted = 0
    total = len(TEST_CASES)

    print(f"\n{'='*88}")
    print(f"{'BM25  vs  BM25+JACCARD  COMPARISON  (top-' + str(K) + ')':^88}")
    print(f"{'='*88}")

    for query, expected, case_k, patterns in TEST_CASES:
        k = max(K, case_k)
        title = _case_id(query)
        qfp = _fp(patterns)
        pat_tokens = sorted(qfp)

        bm25_res = _names(retriever_bm25.retrieve(query, k=k))[:K]
        str_res  = _names(retriever_bm25.retrieve(query, k=k, modeling_patterns=qfp))[:K]

        bm25_hit = any(n in bm25_res for n in expected)
        str_hit  = any(n in str_res  for n in expected)

        activated = len(qfp) >= 2
        lifted    = (not bm25_hit) and str_hit

        if bm25_hit:  hits["BM25"] += 1
        if str_hit:   hits["BM25+Jaccard"] += 1
        if activated: jaccard_activated += 1
        if lifted:    jaccard_lifted += 1

        mark = lambda h: "HIT " if h else "MISS"
        changed = " ← LIFT" if lifted else (" ← REGRESSION" if bm25_hit and not str_hit else "")

        print(f"\n  {title}")
        print(f"  Expected     : {expected}")
        print(f"  Patterns({len(qfp):2d}) : {[t[4:] for t in pat_tokens] if pat_tokens else '(none — Jaccard inactive)'}")
        print(f"  BM25         [{mark(bm25_hit)}]: {bm25_res}")
        print(f"  BM25+Jaccard [{mark(str_hit)}]: {str_res}{changed}")

    print(f"\n{'-'*88}")
    print(f"  Jaccard channel activated (≥2 patterns) in {jaccard_activated}/{total} cases")
    print(f"  Jaccard lifted a miss→hit in {jaccard_lifted} case(s)")
    print(f"\n  Hit rates (top-{K} of {total} cases):")
    for method, count in hits.items():
        bar = "#" * count + "-" * (total - count)
        print(f"  {method:<14} {count}/{total}  [{bar}]  {count/total:.0%}")
    print(f"{'='*88}")
    assert True


@pytest.mark.hybrid
def test_retrieval_method_comparison(retriever_bm25, retriever_hybrid):
    """Side-by-side hit/miss table for BM25, semantic-only, and hybrid.

    All TEST_CASES are shown. The last three (paraphrase queries) reveal
    where semantic finds the exact target while BM25 settles for a related one.
    Run with -s to see the full report.
    """
    K = 3

    hits: dict[str, int] = {"BM25": 0, "Semantic": 0, "Structural": 0, "Hybrid": 0}
    total = len(TEST_CASES)

    print(f"\n{'='*84}")
    print(f"{'RETRIEVAL METHOD COMPARISON  (top-' + str(K) + ')':^84}")
    print(f"{'='*84}")

    for query, expected, case_k, patterns in TEST_CASES:
        k = max(K, case_k)
        title = _case_id(query)
        qfp = _fp(patterns)
        bm25_res = _names(retriever_bm25.retrieve(query, k=k))[:K]
        sem_res  = _names(retriever_hybrid.retrieve(query, k=k, semantic_only=True))[:K]
        str_res  = _names(retriever_bm25.retrieve(query, k=k, modeling_patterns=qfp))[:K]
        hyb_res  = _names(retriever_hybrid.retrieve(query, k=k, modeling_patterns=qfp))[:K]
        bm25_hit = any(n in bm25_res for n in expected)
        sem_hit  = any(n in sem_res  for n in expected)
        str_hit  = any(n in str_res  for n in expected)
        hyb_hit  = any(n in hyb_res  for n in expected)
        if bm25_hit: hits["BM25"] += 1
        if sem_hit:  hits["Semantic"] += 1
        if str_hit:  hits["Structural"] += 1
        if hyb_hit:  hits["Hybrid"] += 1
        mark = lambda h: "HIT " if h else "MISS"
        print(f"\n  {title}")
        print(f"  Expected   : {expected}")
        print(f"  BM25       [{mark(bm25_hit)}]: {bm25_res}")
        print(f"  Semantic   [{mark(sem_hit)}]: {sem_res}")
        print(f"  Structural [{mark(str_hit)}]: {str_res}")
        print(f"  Hybrid     [{mark(hyb_hit)}]: {hyb_res}")

    print(f"\n{'-'*84}")
    print(f"  Hit rates (top-{K} of {total} cases):")
    for method, count in hits.items():
        bar = "#" * count + "-" * (total - count)
        print(f"  {method:<12} {count}/{total}  [{bar}]  {count/total:.0%}")
    print(f"{'='*84}")
    assert True
