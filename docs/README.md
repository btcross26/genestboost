# Optuna-Based Fraud Rule Optimization: Design Journey

## Problem Statement

We want to select and tune a discrete set of fraud detection rules from a large candidate pool. Rules are a mix of continuous-threshold rules (e.g., "model score > t") and purely binary rules (e.g., "transaction is international"). The decisioning logic is OR — if any single active rule fires, the transaction is flagged for review.

The objectives and constraints are operational:

- **Primary objective:** Maximize dollar recall (fraud dollars caught / total fraud dollars)
- **Soft constraint:** Maintain a minimum case recall floor (don't ignore small-dollar fraud entirely)
- **Hard constraint:** Total alert volume must stay within an ops-defined budget (ops reviews every flag regardless of TP/FP)
- **Preference:** Fewer active rules are better for maintainability, but this is a tiebreaker, not a primary objective

---

## Key Design Decisions and Rationale

### Why Optuna Over a Neural Net

The objectives are non-differentiable by nature. Dollar recall is a sum over a thresholded binary mask. OR logic across rules is a step function. Alert count is discrete. A neural net approach would require surrogate losses for all of these (sigmoid approximations, soft-top-k relaxations, differentiable recall proxies), each introducing a gap between what you optimize and what you care about.

Optuna advantages for this problem:

- **Direct metric evaluation.** `flagged[y==1].sum()` is the real metric, not an approximation.
- **Constraints are just code.** Alert budgets, case recall floors, rule exclusion logic — all trivial if-statements, no Lagrangian multipliers.
- **Cross-validation is a loop.** Every trial can be cross-validated inside the objective. You can penalize high variance directly.
- **Pareto front is built-in.** Multi-objective via NSGA-II without scalarization weight tuning.
- **Interpretability.** Every trial has explicit parameter values you can inspect and explain to stakeholders.

A neural net wins if you have 500+ candidate rules, need to learn rules (not just select), or have massive datasets requiring gradient efficiency. For 30-50 candidate rules with known definitions, black-box optimization is the right tool.

### Why CMA-ES as the Sampler

CMA-ES maintains a full covariance matrix across all parameters. It naturally learns joint structure like "loosening rule A makes rule B redundant." TPE (Optuna's default) models each parameter's marginal independently, missing these cross-parameter dependencies.

CMA-ES is ideal when:

- The search space is continuous (or can be made continuous via the sigmoid trick)
- Dimensionality is moderate (30-50 parameters works well, past 100 it struggles)
- Parameter correlations matter (they do — rules interact through the shared alert budget)

**Caveat:** CMA-ES is fundamentally a local optimizer. It converges to one basin. For multimodal landscapes (multiple very different rule combinations achieving similar performance), use the IPOP restart strategy or a two-phase TPE → CMA-ES approach (detailed below).

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────┐
│  Stage 1: Rule Generation                           │
│  Fit shallow trees (depth 1, 2, 3) + extract GBM   │
│  tree paths → hundreds of candidate rules           │
├─────────────────────────────────────────────────────┤
│  Stage 2: Pre-Filtering                             │
│  Evaluate on holdout: fire rate, precision, dollar  │
│  precision, minimum support thresholds              │
├─────────────────────────────────────────────────────┤
│  Stage 3: Redundancy Removal                        │
│  Cluster by Jaccard similarity on firing patterns   │
│  Select diverse, high-quality reps per cluster      │
│  → ~30-50 genuinely distinct candidates             │
├─────────────────────────────────────────────────────┤
│  Stage 4: Optuna Joint Optimization                 │
│  CMA-ES over gate logits + continuous thresholds    │
│  Maximize dollar recall under alert + case recall   │
│  constraints                                        │
├─────────────────────────────────────────────────────┤
│  Stage 5: Selection and Validation                  │
│  Filter to feasible trials, validate on held-out    │
│  data, inspect Pareto front if multi-objective      │
└─────────────────────────────────────────────────────┘
```

---

## Stage 1: Rule Candidate Generation

Generate rules at multiple depths for different complexity levels. Depth 1 gives single-split rules ("amount > 5000"), depth 2 gives two-condition conjunctions, depth 3 gives three-condition conjunctions.

```python
from sklearn.tree import DecisionTreeClassifier
import numpy as np

candidates = []

for depth in [1, 2, 3]:
    for _ in range(500):
        dt = DecisionTreeClassifier(
            max_depth=depth,
            max_features="sqrt",
            min_samples_leaf=50,
        )
        idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
        dt.fit(X_train[idx], y_train[idx])
        candidates.extend(extract_leaf_rules(dt, feature_names))
```

GBM trees as an additional source — each successive tree fits residuals, so they naturally find incremental signal:

```python
import xgboost as xgb

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300)
gbm.fit(X_train, y_train)

for tree_idx in range(gbm.n_estimators):
    candidates.extend(extract_leaf_rules_from_booster(gbm, tree_idx, feature_names))
```

Each candidate rule is a callable that takes a dataset and returns a boolean firing vector. Rules with continuous thresholds store their threshold as a tunable parameter. Binary rules (e.g., "is_international") have no threshold.

### Feature Binning Option

Binning continuous features before tree fitting produces cleaner, more repeatable split points and reduces near-duplicate rules. This is a preprocessing decision that trades some granularity for much less deduplication overhead downstream.

---

## Stage 2: Pre-Filtering on Eval Set

Eliminate noise rules before they reach the optimizer. Each candidate is evaluated on a holdout eval set.

```python
rule_stats = []
for rule in candidates:
    fires = rule.evaluate(X_eval)
    fire_rate = fires.mean()
    n_fires = fires.sum()
    precision = y_eval[fires].mean() if fires.any() else 0
    dollar_precision = (
        fraud_dollars[fires & (y_eval == 1)].sum() / n_fires
        if n_fires > 0 else 0
    )

    rule_stats.append({
        "rule": rule,
        "fire_rate": fire_rate,
        "precision": precision,
        "dollar_precision": dollar_precision,
        "n_fires": n_fires,
    })

filtered = [
    r for r in rule_stats
    if r["fire_rate"] >= 0.001       # fires on at least 0.1% of traffic
    and r["precision"] >= 0.05       # at least 5% hit rate
    and r["n_fires"] >= min_support  # enough volume to trust the estimate
]

filtered.sort(key=lambda r: r["dollar_precision"], reverse=True)
```

---

## Stage 3: Redundancy Removal via Firing-Pattern Clustering

Two rules can look completely different syntactically but fire on the same transactions. Deduplication must operate on firing patterns, not rule definitions.

### Clustering

```python
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

# build binary firing matrix: rows = observations, cols = rules
fire_matrix = np.column_stack([
    r["rule"].evaluate(X_eval).astype(float) for r in filtered
])

# jaccard distance between firing patterns
distances = pdist(fire_matrix.T, metric="jaccard")
Z = linkage(distances, method="complete")

# 0.3 distance threshold = rules sharing 70%+ of flags collapse together
clusters = fcluster(Z, t=0.3, criterion="distance")
```

### Within-Cluster Selection via Maximal Marginal Relevance

Don't just take top-1 per cluster. Some clusters contain rules with meaningful 30% divergence worth preserving.

```python
def jaccard_similarity(a, b):
    intersection = (a & b).sum()
    union = (a | b).sum()
    return intersection / union if union > 0 else 0.0

def select_from_cluster(cluster_rules, cluster_fire_matrix, k=3, min_score=0.01):
    """Pick up to k rules: best first, then greedily by marginal diversity."""
    scores = [r.dollar_precision for r in cluster_rules]
    selected = [int(np.argmax(scores))]

    for _ in range(k - 1):
        best_next, best_score = None, -1
        for j in range(len(cluster_rules)):
            if j in selected:
                continue
            # max overlap with any already-selected rule
            max_overlap = max(
                jaccard_similarity(
                    cluster_fire_matrix[:, j].astype(bool),
                    cluster_fire_matrix[:, s].astype(bool),
                )
                for s in selected
            )
            diversity = 1 - max_overlap
            quality = cluster_rules[j].dollar_precision
            score = quality * diversity

            if score > best_score:
                best_score = score
                best_next = j

        if best_next is None or best_score < min_score:
            break
        selected.append(best_next)

    return [cluster_rules[i] for i in selected]


final_candidates = []
for c in np.unique(clusters):
    mask = clusters == c
    cluster_rules = [filtered[i]["rule"] for i in range(len(filtered)) if mask[i]]
    cluster_fires = fire_matrix[:, mask]
    final_candidates.extend(select_from_cluster(cluster_rules, cluster_fires, k=3))

print(f"Final candidate pool: {len(final_candidates)} rules")
```

The `quality * diversity` product implements maximal marginal relevance: a rule needs to be both good on its own and different from what's already selected. The Jaccard threshold (0.3) and within-cluster k (3) are tunable — lower threshold or lower k gives a sparser, more diverse pool; higher values preserve more candidates for the optimizer to sort out.

---

## Stage 4: Optuna Optimization

### Parameterization: The Sigmoid Gate Trick

The central insight: replace binary on/off indicators with continuous logits passed through a sigmoid. This gives CMA-ES a smooth landscape instead of a combinatorial one.

- **Continuous-threshold rules:** The threshold parameter IS the gate. Push it to an extreme where the rule never fires and it's effectively off.
- **Binary rules (no threshold):** Need a separate continuous logit. `gate > 0` means on, `gate < 0` means off. CMA-ES learns to push these to extremes.

```python
import optuna
import math

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

sampler = optuna.samplers.CmaEsSampler(
    restart_strategy="ipop",  # restart with increasing population for multimodality
    n_startup_trials=20,
)
study = optuna.create_study(direction="maximize", sampler=sampler)

def objective(trial):
    flagged = np.zeros(len(X_test), dtype=bool)

    for i, rule in enumerate(final_candidates):
        if rule.has_threshold:
            t = trial.suggest_float(rule.name, rule.lo, rule.hi)
            fires = rule.evaluate(X_test, t)
        else:
            gate = trial.suggest_float(f"gate_{rule.name}", -6.0, 6.0)
            fires = rule.evaluate(X_test) if gate > 0 else np.zeros(len(X_test), dtype=bool)

        flagged |= fires

    # metrics
    total_alerts = flagged.sum()
    dollar_recall = (
        fraud_dollars[flagged & (y_test == 1)].sum()
        / fraud_dollars[y_test == 1].sum()
    )
    case_recall = flagged[y_test == 1].mean()
    n_active = sum(
        1 for i, r in enumerate(final_candidates)
        if r.has_threshold or trial.params.get(f"gate_{r.name}", -1) > 0
    )

    # smooth penalties instead of hard cliffs
    alert_overshoot = max(0, total_alerts - alert_budget) / alert_budget
    case_shortfall = max(0, min_case_recall - case_recall)
    penalty = 100 * alert_overshoot ** 2 + 100 * case_shortfall ** 2

    # optional: mild tiebreaker for fewer rules
    sparsity = 0.001 * n_active

    # stash real metrics for post-hoc selection
    trial.set_user_attr("dollar_recall", dollar_recall)
    trial.set_user_attr("case_recall", case_recall)
    trial.set_user_attr("total_alerts", int(total_alerts))
    trial.set_user_attr("n_active", n_active)

    return dollar_recall - penalty - sparsity

study.optimize(objective, n_trials=500)
```

### Why Smooth Penalties Instead of Hard Cliffs

Hard cliffs (`return -1e6 if over budget`) waste trials because CMA-ES gets zero signal about how far over budget a solution is. A quadratic ramp tells the sampler which direction to move — 1% over budget is a nudge, 50% over is a wall. The optimizer finds the feasible region faster.

Normalizing alert overshoot by the budget keeps penalty scale consistent regardless of absolute budget size.

### Cross-Validation Variant

Every trial can be cross-validated to favor rule sets that generalize:

```python
def objective_cv(trial):
    fold_metrics = []
    for train_idx, val_idx in kfold.split(X):
        # ... suggest params (same across folds) ...
        flagged = evaluate_rules(X[val_idx], trial)
        dr = fraud_dollars[val_idx][flagged & (y[val_idx] == 1)].sum() / fraud_dollars[val_idx][y[val_idx] == 1].sum()
        fold_metrics.append(dr)

    mean_dr = np.mean(fold_metrics)
    std_dr = np.std(fold_metrics)

    trial.set_user_attr("cv_std", std_dr)

    # penalize variance — favor stable rule sets
    return mean_dr - 0.5 * std_dr
```

---

## Stage 5: Selection and Validation

### Post-Hoc Feasibility Filtering

Smooth penalties guide the search but don't guarantee hard constraints. Filter after optimization:

```python
feasible = [
    t for t in study.trials
    if t.user_attrs["total_alerts"] <= alert_budget
    and t.user_attrs["case_recall"] >= min_case_recall
]

best = max(feasible, key=lambda t: t.user_attrs["dollar_recall"])

# extract the final rule set
active_rules = []
for i, rule in enumerate(final_candidates):
    if rule.has_threshold:
        # check if threshold produces any fires
        t = best.params[rule.name]
        if rule.evaluate(X_test, t).any():
            active_rules.append((rule, t))
    else:
        if best.params.get(f"gate_{rule.name}", -1) > 0:
            active_rules.append((rule, None))

print(f"Final rule set: {len(active_rules)} rules")
print(f"Dollar recall: {best.user_attrs['dollar_recall']:.4f}")
print(f"Case recall:   {best.user_attrs['case_recall']:.4f}")
print(f"Total alerts:  {best.user_attrs['total_alerts']}")
```

### Multi-Objective Pareto Front (Alternative)

If you want to explore the tradeoff between dollar recall and case recall rather than fixing a floor:

```python
study = optuna.create_study(
    directions=["maximize", "maximize"],
    sampler=optuna.samplers.NSGAIISampler(),  # CMA-ES doesn't support multi-objective
)

def objective_mo(trial):
    # ... same rule logic ...

    if total_alerts > alert_budget:
        return -1e6, -1e6

    return dollar_recall, case_recall

study.optimize(objective_mo, n_trials=500)

# study.best_trials is the Pareto set — no single winner
# plot and discuss with ops to pick a point on the frontier
pareto = study.best_trials
for t in pareto:
    print(f"DR={t.values[0]:.3f}  CR={t.values[1]:.3f}  "
          f"alerts={t.user_attrs['total_alerts']}")
```

The Pareto front is most valuable as a communication tool. Bring the plot to stakeholders: "we can catch 90% of fraud dollars at 85% case recall and 3,000 alerts/day, or 82% of fraud dollars at 92% case recall and 1,800 alerts — where do you want to live?"

**Note:** NSGA-II is the multi-objective sampler. CMA-ES in Optuna doesn't natively support multiple objectives. You lose covariance learning but gain the full Pareto front.

---

## CMA-ES Mechanics Reference

CMA-ES maintains three pieces of state: a mean vector μ (where to search), a covariance matrix C (the shape of the search cloud), and a step size σ (how far to reach).

**Each iteration:**

1. **Sample.** Draw λ candidates from N(μ, σ²C).
2. **Evaluate.** Run the objective on each candidate.
3. **Rank.** Sort by fitness, weight the top ~λ/2.
4. **Update mean.** Shift μ toward the weighted centroid of top candidates.
5. **Update covariance.** C adapts via two mechanisms:
   - *Rank-one update:* the evolution path (exponential moving average of mean shifts) is injected as an outer product. If the mean keeps moving the same direction, C elongates along that axis.
   - *Rank-μ update:* the spread of the current top candidates is folded into C. If winners are spread along a diagonal, C stretches to match.
6. **Update step size.** σ adapts via cumulative step-size adaptation. Correlated successive steps → σ grows (steps too short). Canceling steps → σ shrinks (overshooting).

Over time, C learns the correlation structure of your problem. Early iterations sample from a rough sphere. After convergence, the search distribution is a thin, tilted ellipsoid aligned with the landscape — it knows which parameters travel together in good solutions.

### Multimodality and Exploration

CMA-ES is a local optimizer. It converges to one basin. For multimodal landscapes (multiple distinct high-performing rule sets), use:

- **IPOP restarts:** Run until σ collapses, restart with fresh mean, reset C, double population size. Each restart is an independent shot at a different basin.
- **TPE → CMA-ES phasing:** Use TPE for the first 100-200 trials (naturally scatters across modes), then switch to CMA-ES to refine the best region.

```python
# IPOP restarts
sampler = optuna.samplers.CmaEsSampler(restart_strategy="ipop")

# or two-phase approach
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=150)  # TPE phase

study.sampler = optuna.samplers.CmaEsSampler(restart_strategy="ipop")
study.optimize(objective, n_trials=350)  # CMA-ES refinement phase
```

After optimization, inspect top trials for basin diversity: if the top 10 trials show two or three distinct rule sets, CMA-ES found multiple modes across restarts. If they're all minor variations, the landscape is effectively unimodal.

---

## Open Design Questions

### Soft Relaxation vs. Direct Binary

We explored a sigmoid relaxation approach where gate logits are continuous and weights are soft during optimization. The simplified OR-logic formulation may not need this — the gate logit with a hard threshold at zero is sufficient because CMA-ES handles the step function adequately. The soft relaxation is most valuable if:

- You want to do staged pruning (iterative rounds with increasing pressure)
- You need smoother signal during early exploration
- The number of candidates is large enough that CMA-ES needs help

For 30-50 candidates with OR logic, the hard-gate formulation is likely sufficient.

### Staged Pruning vs. Single-Shot

We discussed an iterative approach: run optimization with soft gates and low ambiguity penalty, prune low-weight rules, tighten the penalty, repeat. This gives inspection points between rounds and a shrinking search space. The tradeoff is complexity — single-shot CMA-ES with IPOP restarts may find the same solutions with less machinery.

Staged pruning is worth it if:

- You start with 100+ candidates (too many for CMA-ES in one shot)
- You want human-in-the-loop inspection between rounds
- Domain knowledge suggests certain rules should be evaluated together before pruning

### Rule Count Control

Options explored, from softest to hardest:

- **Implicit via alert budget:** More active rules cost alert volume. The budget naturally limits rule count without explicit penalization.
- **Mild tiebreaker:** `0.001 * n_active` — between equal solutions, prefer simpler.
- **Quadratic knee:** `λ * max(0, n_active - K)²` — don't care up to K rules, steep penalty beyond.
- **Hard cliff:** `return -1e6 if n_active > K` — strict cap, wastes trials near the boundary.

The alert budget alone may be sufficient. Each active rule burns alert capacity, so the optimizer only keeps rules that earn their keep.

### Number of Rules to Carry Into Optuna

The firing-pattern clustering with MMR selection should produce a pool where every candidate represents genuinely different fraud signal. Target ~30-50 for CMA-ES comfort. If clustering produces more, tighten the Jaccard threshold or reduce within-cluster k. If it produces fewer, that's fine — the optimizer has less work to do.

### Pareto Front vs. Constrained Single-Objective

Run the Pareto front (dollar recall vs. case recall) at least once to understand the tradeoff shape. If the two objectives are highly correlated across the frontier, the case recall floor is unnecessary. If there's a sharp elbow, that tells you exactly where to set `min_case_recall` for subsequent constrained single-objective runs with CMA-# MILP for Fraud Rule Optimization

## What is MILP?

Mixed-Integer Linear Programming (MILP) is an optimization framework where you minimize or maximize a linear objective function, subject to linear constraints, with the requirement that some or all decision variables must take integer values (often binary 0/1). The solver explores feasible combinations of integer variables using branch-and-bound, cutting planes, and LP relaxations — brute-force search made tractable.

**Tooling:** Google's open-source OR-Tools library (`ortools.linear_solver.pywraplp`) supports SCIP as a backend solver for MILP problems.

---

## The Problem

Given a dataset with:

- A **ground truth binary column** (e.g., confirmed fraud)
- **Hundreds of candidate binary rule columns** (e.g., alert rules)

Find the smallest or least-complex subset of rules such that their **OR** (at least one fires) achieves a minimum hit rate on the ground truth, subject to operational constraints like a cap on total alert volume.

This is a **weighted set cover** problem — covering the positive-label rows with the fewest (or cheapest) rule columns.

---

## Core Formulation

### Decision Variables

| Variable | Meaning |
|----------|---------|
| $z_j \in \{0,1\}$ | Whether candidate rule $j$ is selected |
| $f_i \in \{0,1\}$ | Whether row $i$ is "fired on" (at least one selected rule hits) |

The $z_j$ variables are the true decisions. The $f_i$ variables are mechanical bookkeeping — fully determined by the $z_j$ choices — needed only to express the hit rate and volume constraints linearly.

### Linearizing the OR Condition

For the OR of selected rules, the only constraints needed are:

$$f_i \geq z_j \quad \text{for all } (i, j) \text{ where } x_{ij} = 1$$

This says: if rule $j$ is selected ($z_j = 1$) and rule $j$ fires on row $i$ ($x_{ij} = 1$), then row $i$ must be flagged ($f_i = 1$).

No upper-bound constraint on $f_i$ is needed. Since $f_i \in \{0,1\}$ by its domain and $f_i$ is not rewarded in the objective, the solver has no incentive to push it to 1 unless forced by some active $z_j$.

The constraint set is extremely sparse — you only generate entries where rules actually fire.

### Objective: Minimize Number of Rules

$$\text{minimize} \sum_j z_j$$

### Constraint: Minimum Recall (Hit Rate on Positives)

$$\sum_{i:\, y_i = 1} f_i \;\geq\; \tau \cdot N^+$$

where $\tau$ is the target recall and $N^+$ is the total number of positive-label rows.

### Constraint: Maximum Alert Volume

$$\sum_i f_i \;\leq\; M$$

where $M$ is the maximum number of rows allowed to fire.

---

## Adding Rule Complexity

Instead of treating all rules equally, assign a complexity cost $c_j$ to each rule (e.g., 1, 2, 10). The objective becomes:

$$\text{minimize} \sum_j c_j \cdot z_j$$

This is the general **weighted set cover**. Minimizing number of rules is the special case where all $c_j = 1$.

You can also combine both — minimize total complexity while capping the number of rules:

$$\text{minimize} \sum_j c_j \cdot z_j$$

$$\sum_j z_j \leq K$$

This says: among all feasible sets of $\leq K$ rules meeting recall and volume constraints, find the one with lowest total complexity.

---

## Full Formulation (with Aggregation)

### Variables

- $z_j \in \{0,1\}$ for each candidate rule $j$
- $f_k \in \{0,1\}$ for each distinct row-pattern group $k$

### Objective

$$\text{minimize} \sum_j c_j \cdot z_j$$

### Constraints

| Constraint | Expression | Purpose |
|------------|------------|---------|
| OR linkage | $f_k \geq z_j$ for all $(k,j)$ where $x_{kj} = 1$ | If a selected rule fires on pattern $k$, flag it |
| Minimum recall | $\sum_k w_k^+ \cdot f_k \geq \tau \cdot N^+$ | Cover enough fraud |
| Maximum volume | $\sum_k w_k \cdot f_k \leq M$ | Don't blow up the alert queue |
| Max rules (optional) | $\sum_j z_j \leq K$ | Limit number of active rules |

Where:

- $w_k$ = total number of rows in pattern group $k$
- $w_k^+$ = number of positive-label rows in pattern group $k$
- $N^+$ = total number of positive-label rows

---

## Row Aggregation

### The Principle

Rows with identical binary patterns across all candidate rule columns are indistinguishable from the solver's perspective. They can be collapsed into a single representative with a weight. This is exact — not an approximation.

This technique is standard in operations research (called "row aggregation," "demand aggregation," or "scenario reduction" depending on the domain).

### Why It Works Here

1. **Drop non-firing rows entirely.** Rows where no candidate rule fires have $f_k = 0$ guaranteed. They contribute nothing to any constraint.
2. **Collapse among firing rows.** If you have 100k rows where at least one rule fires, the number of distinct binary patterns is typically far smaller — often a few thousand — because fraud rules are correlated and sparse.
3. **The aggregation is valid as long as grouped rows have identical coefficients in every constraint.** If you later add constraints that differentiate rows within a group (e.g., per-customer caps, time-windowed limits), you'd need to re-partition.

### Example

| Scenario | Row Count |
|----------|-----------|
| Total transactions | 5,000,000 |
| Rows where ≥1 rule fires | 100,000 |
| Distinct binary patterns | ~2,000–5,000 |

The solver operates on the ~2,000–5,000 pattern groups, not millions of rows.

---

## Exploring the Pareto Frontier (Recall vs. Complexity)

The tradeoff between recall and complexity can be mapped by sweeping one parameter while optimizing the other.

### Approach 1: Sweep the Recall Threshold

Solve the MILP at $\tau = 0.95, 0.90, 0.85, \ldots$ and collect the optimal complexity at each level:

$$\text{minimize} \sum_j c_j \cdot z_j \quad \text{subject to recall} \geq \tau$$

### Approach 2: Sweep the Complexity Budget (Flipped Formulation)

Put recall in the objective and complexity in the constraint:

$$\text{maximize} \sum_k w_k^+ \cdot f_k$$

$$\sum_j c_j \cdot z_j \leq C$$

Then sweep $C$ upward. At each budget level you get the best recall achievable.

### Flipping is General

These two approaches trace the same Pareto frontier from different axes. This is a general principle in optimization: "minimize A subject to B ≥ threshold" and "maximize B subject to A ≤ budget" yield the same set of Pareto-optimal points. The aggregation and constraint structure remain identical — you're just reassigning which linear expression is the objective and which is a constraint.

The two formulations may have different computational performance (LP relaxation tightness, branching behavior), so if one direction is slow, it's worth trying the flip.

### The Typical Shape

The frontier usually has a clear **elbow** — you get a lot of recall cheaply with the first few rules, then it gets expensive fast. That elbow is the natural operating point and often the most valuable output for stakeholder conversations.

---

## Infeasibility

MILP solvers return a status flag indicating whether a feasible solution exists:

```python
status = solver.Solve()
if status == pywraplp.Solver.INFEASIBLE:
    # no feasible solution exists
```

Infeasibility is common and informative. You could easily create contradictory demands — "achieve 95% recall with at most 3 rules and cap alerts at 500" — where no combination of rules satisfies all constraints simultaneously.

### Diagnostic Use

Relax one constraint at a time to find the binding bottleneck:

- Is it the recall floor?
- The volume cap?
- The max number of rules?
- The complexity budget?

This conversation with stakeholders is often more valuable than the solution itself.

### Why Infeasibility Is More Common in MILP Than Continuous LP

In continuous optimization (e.g., structural engineering), the feasible region typically has a continuous interior — you can always find some feasible point even if it's at a boundary. With integer constraints, you're restricted to lattice points, and the feasible set can easily be empty. There's no continuous interior to fall back on.

---

## Summary of Key Insights

- **OR linearization** is simpler than AND — you only need $f_i \geq z_j$ where $x_{ij} = 1$
- **No upper bound on $f_i$ needed** — domain ($\{0,1\}$) and lack of objective incentive handle it
- **Weighted set cover** generalizes naturally from "fewest rules" to "least complex rules"
- **Stacking business constraints** (recall floor, volume cap, rule count limit, complexity budget) is just adding rows to the formulation — no algorithmic redesign
- **Row aggregation** reduces millions of rows to thousands of pattern groups exactly
- **Pareto frontier** exploration is a simple loop of MILP solves, and the objective/constraint roles can be flipped
- **Infeasibility is a feature** — it reveals which business requirements conflict




