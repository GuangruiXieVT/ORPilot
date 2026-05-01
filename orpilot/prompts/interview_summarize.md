---
version: 1.0.0
---

Based on the following conversation, extract a structured problem definition.

Conversation:
{conversation}

Provide:
- title: short problem title
- description: full natural language description
- problem_type: one of linear_programming, integer_programming, mixed_integer, transportation, assignment, scheduling, network_flow, other
- objective: minimize or maximize
- objective_description: what is being optimized
- constraints: list of constraints (description + mathematical expression if clear)
- decision_variables: list of variable descriptions
- parameters: list of parameter descriptions, including their indices (e.g. "cost[i,j] = cost of shipping from i to j")
- additional_notes: anything else relevant

CRITICAL RULE when writing constraints, decision_variables, and parameters:
Never merge distinct entity types into a single combined index. Each entity type that has a different operational role must keep its own dedicated index symbol throughout.

WRONG — merging production sites and distribution centers into one "Locations" index:
  storage_capacity[l] for all l in Locations (= production sites ∪ DCs)
  inventory[l,p,t] for all l in Locations
This forces a single combined entity that cannot be cleanly separated later.

CORRECT — keep each entity type as its own distinct index:
  storage_capacity_site[i] for all i in ProductionSites
  storage_capacity_dc[j]   for all j in DistributionCenters
  inventory_site[i,p,t]    for all i in ProductionSites
  inventory_dc[j,p,t]      for all j in DistributionCenters

Apply this to every parameter and variable: if a parameter applies to both production sites AND distribution centers, write it twice — once per entity type — with separate index symbols. Never collapse them into a single "facility" or "location" index. This includes fixed opening costs and operating costs — NEVER write fixed_cost[facility_type, facility_id]; always write fixed_opening_cost_site[i] and fixed_opening_cost_dc[j] separately.

EXCEPTION: In routing/VRP problems, merging the depot and customers into a single Locations set is standard and explicitly allowed — arc variables and distance/cost parameters indexed over Locations × Locations are correct in that context.
