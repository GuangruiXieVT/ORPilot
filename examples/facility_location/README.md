# Facility Location Problem

Select which warehouses to open and assign customers to open warehouses to minimize total fixed plus shipping cost.

## Problem

- **Decision variables:**
  - `open[w]` — 1 if warehouse `w` is opened, 0 otherwise (binary)
  - `assign[w, c]` — 1 if customer `c` is served by warehouse `w`, 0 otherwise (binary)
- **Objective:** Minimize total cost = Σ fixed_cost[w] × open[w] + Σ shipping_cost[w,c] × demand[c] × assign[w,c]
- **Constraints:**
  - Each customer assigned to exactly one warehouse
  - Customers can only be assigned to open warehouses
  - Total demand served by each warehouse ≤ its capacity

## Data

| File | Description |
|------|-------------|
| `data/warehouses.csv` | Warehouse IDs, fixed opening costs, and capacities |
| `data/customers.csv` | Customer IDs and demand requirements |
| `data/shipping.csv` | Per-unit shipping cost from each warehouse to each customer |

### warehouses.csv
| warehouse_id | fixed_cost | capacity |
|---|---|---|
| W1 | 1000 | 200 |
| W2 | 1500 | 150 |
| W3 | 800 | 150 |

### customers.csv
| customer_id | demand |
|---|---|
| C1 | 50 |
| C2 | 80 |
| C3 | 60 |
| C4 | 70 |

### shipping.csv (unit shipping cost)
| warehouse_id | customer_id | cost |
|---|---|---|
| W1 | C1 | 2 |
| W1 | C2 | 3 |
| W1 | C3 | 5 |
| W1 | C4 | 4 |
| W2 | C1 | 4 |
| W2 | C2 | 1 |
| W2 | C3 | 3 |
| W2 | C4 | 2 |
| W3 | C1 | 5 |
| W3 | C2 | 4 |
| W3 | C3 | 1 |
| W3 | C4 | 2 |

## Expected Optimal Solution

**Objective value: $2340**

Open W1 and W3 (fixed cost: $1000 + $800 = $1800):

| Assignment | Units | Unit Cost | Subtotal |
|------------|-------|-----------|----------|
| W1 → C1 | 50 | $2 | $100 |
| W1 → C2 | 80 | $3 | $240 |
| W3 → C3 | 60 | $1 | $60 |
| W3 → C4 | 70 | $2 | $140 |
| **Total** | | | **$540** |

Fixed cost $1800 + Shipping $540 = **$2340**
