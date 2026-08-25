# QBoost Design

Formulation, and an explicit statement of which parts are quantum and which
are not. Written **before** implementation so the honesty claims are not
retrofitted to whatever the results turned out to be.

---

## 1. What QBoost is

QBoost (Neven et al., 2008–2012) recasts ensemble selection as a binary
optimisation problem. Instead of greedily adding weak learners the way AdaBoost
does, it asks: *which subset of a fixed pool of weak learners minimises
training error when their votes are summed?* That subset-selection problem is
naturally a QUBO, which is why it became a canonical quantum-annealing
benchmark.

The quantum content is entirely in **how the QUBO is solved**. The boosting
formulation itself is classical.

## 2. Objective

Given `N` weak learners `h_i(x) ∈ {-1, +1}`, `M` training samples with labels
`y_m ∈ {-1, +1}`, and binary selection variables `w_i ∈ {0, 1}`:

```
minimise   sum_m ( (1/N) sum_i w_i h_i(x_m) - y_m )^2   +   lambda * sum_i w_i
             \_______________ squared ensemble error ______/     \__ sparsity __/
```

Expanding the square and dropping the `w`-independent constant `sum_m y_m^2`:

```
sum_m ( (1/N^2) sum_ij w_i w_j h_i(x_m) h_j(x_m)  -  (2/N) sum_i w_i h_i(x_m) y_m )
```

which gives the standard QUBO form `w^T Q w` with

```
Q_ij  =  (1/N^2) * sum_m h_i(x_m) h_j(x_m)                 for i != j
Q_ii  =  (1/N^2) * sum_m h_i(x_m)^2
       -  (2/N)  * sum_m h_i(x_m) y_m
       +  lambda
```

`Q_ij` is the **correlation between two learners' predictions** — QBoost
penalises selecting redundant learners. `Q_ii` rewards a learner that
correlates with the label, and `lambda` charges a fixed price per selected
learner, controlling ensemble size.

## 3. Weak learners

Depth-1 decision stumps, `n_weak_learners = 16` (config). Stumps are the
classical QBoost choice: individually barely better than chance, so the
*selection* does the work. Hundreds of learners would make the QUBO too large
to simulate on a gate model, which would defeat the comparison.

Each stump is fitted on the controlled training fold. The QUBO is built from
predictions on the **training** fold only.

## 4. Solvers, and what is actually quantum

Two solvers are implemented. Both are reported separately and neither is
described as something it is not.

### 4.1 `simulated_annealing` — **CLASSICAL**

A classical Metropolis simulated-annealing sweep over the binary vector.
This is the default because it scales to any QUBO size here.

> This is **not** quantum execution. It is a classical heuristic solving a
> problem that *could* be given to a quantum annealer. Reporting it as
> "quantum" would be false. It appears in results as
> `qboost_sa (classical solver)`.

### 4.2 `qaoa` — **GATE-MODEL QUANTUM (simulated)**

The Quantum Approximate Optimisation Algorithm run on the Qiskit statevector
simulator. The QUBO is converted to an Ising Hamiltonian, and QAOA optimises
the variational angles. This *is* a genuine quantum algorithm, executed on a
**simulator**, not on hardware.

One qubit per weak learner, so QAOA is attempted only when
`n_weak_learners <= qaoa_max_variables` (default 12). Beyond that the
statevector is too large to be worth the runtime here.

## 5. Pipeline

```
Controlled training fold
        ↓
16 decision stumps (classical)
        ↓
Prediction matrix H  (M x N, entries in {-1,+1})
        ↓
QUBO construction: Q from H and y   (classical)
        ↓
Solve for w ∈ {0,1}^N       <-- simulated annealing (classical)
                            <-- or QAOA (quantum, simulated)
        ↓
Boosted score(x) = (1/|w|) * sum_i w_i h_i(x)
        ↓
Threshold tuned on VALIDATION
```

The final score is a normalised vote fraction. It is a **score**, not a
calibrated probability, and is reported as such.

## 6. Baselines

Compared on the identical controlled subset:

| Model | Ensemble selection |
|---|---|
| AdaBoost | greedy, re-weighting |
| Gradient Boosting | greedy, gradient steps |
| QBoost (SA solver) | global QUBO subset selection, classical solver |
| QBoost (QAOA solver) | global QUBO subset selection, quantum simulator |

The comparison that matters is **QBoost vs classical boosting on equal
footing** (same stumps, same data), plus **SA vs QAOA** to see whether the
quantum solver finds a different subset at all.

## 7. What this cannot show

- No quantum advantage. A 16-variable QUBO is trivially solvable classically;
  QAOA here is a correctness demonstration, not a performance claim.
- Runtime comparisons between SA and QAOA measure simulator overhead, not
  hardware speed.
- QBoost beating AdaBoost (or not) is a statement about *global vs greedy
  subset selection*, which is a classical algorithmic difference. It says
  nothing about quantum computing either way.
