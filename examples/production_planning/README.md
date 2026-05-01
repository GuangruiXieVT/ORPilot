# Production Planning (Capacitated Lot Sizing)

Minimize total production, setup, and holding costs over a multi-period planning horizon.

## Problem

- **Decision variables:**
  - `produce[t]` — units produced in period `t` (continuous, 0 ≤ produce ≤ capacity)
  - `setup[t]` — 1 if production occurs in period `t`, 0 otherwise (binary)
  - `inventory[t]` — units held at end of period `t` (continuous, ≥ 0)
- **Objective:** Minimize total cost = Σ (variable_cost × produce[t] + setup_cost × setup[t] + holding_cost × inventory[t])
- **Constraints:**
  - Inventory balance: inventory[t] = inventory[t-1] + produce[t] − demand[t]
  - Capacity: produce[t] ≤ max_production × setup[t]
  - Initial and final inventory = 0

## Data

| File | Description |
|------|-------------|
| `data/periods.csv` | Period IDs and demand per period |
| `data/params.csv` | Global parameters (costs, capacity) |

### periods.csv
| period_id | demand |
|-----------|--------|
| 1 | 10 |
| 2 | 20 |
| 3 | 30 |
| 4 | 20 |

### params.csv
| param | value |
|-------|-------|
| variable_cost | 5 |
| setup_cost | 100 |
| holding_cost | 2 |
| max_production | 50 |

## Expected Optimal Solution

**Objective value: $680**

| Period | Produce | Setup | End Inventory | Period Cost |
|--------|---------|-------|---------------|-------------|
| 1 | 30 | 1 | 20 | $250 + $40 |
| 2 | 0 | 0 | 0 | $0 |
| 3 | 50 | 1 | 20 | $350 + $40 |
| 4 | 0 | 0 | 0 | $0 |
| **Total** | **80** | | | **$680** |

Produce in periods 1 and 3 to cover two periods' demand each, minimizing setup costs while staying within the 50-unit capacity limit.
