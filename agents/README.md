# Vega SDLC Agent Framework

This directory defines a lightweight, auditable **Software Development Life Cycle (SDLC) Agent Framework** for the Vega Quant Trading Engine.

---

## 1. Architectural Philosophy

### Why SDLC Automation Instead of Autonomous Trading Agents?

In quantitative finance and algorithmic execution, **runtime determinism, mathematical verifiability, and auditability are non-negotiable**. 

- **Autonomous Runtime Agents (Rejected)**: Placing LLM or generative agents directly inside order execution or strategy decision loops introduces non-deterministic latency, hallucinated order quantities, opaque reasoning, and unacceptable regulatory liability.
- **SDLC Automation Agents (Adopted)**: Generative AI agents are exceptionally well-suited for **automating the engineering lifecycle** around the trading engine:
  1. Translating quantitative requirements into rigorous engineering specifications.
  2. Generating exhaustive edge-case test matrices (boundary conditions, state transitions, failovers).
  3. Enforcing institutional architectural guidelines and lookahead-bias prevention during code review.

```
                     ┌───────────────────────────┐
                     │ Quantitative Change / RFC │
                     └─────────────┬─────────────┘
                                   │
                     ┌─────────────▼─────────────┐
                     │       Planner Agent       │
                     │  - Requirements & Scope   │
                     │  - Architectural Guardrail │
                     │  - Acceptance Criteria    │
                     └─────────────┬─────────────┘
                                   │
                     ┌─────────────▼─────────────┐
                     │      Test / QA Agent      │
                     │  - Boundary Test Matrix   │
                     │  - Failure & State Tests  │
                     │  - Regression Invariants  │
                     └─────────────┬─────────────┘
                                   │
                     ┌─────────────▼─────────────┐
                     │   Developer Implementation│
                     │   (Clean Code + Tests)    │
                     └─────────────┬─────────────┘
                                   │
                     ┌─────────────▼─────────────┐
                     │     Code Review Agent     │
                     │  - Architectural Review   │
                     │  - Lookahead Prevention   │
                     │  - Safety & Gate Decision │
                     └───────────────────────────┘
```

---

## 2. Directory Layout

| Artifact | Role | Purpose |
| :--- | :--- | :--- |
| [`planner.md`](./planner.md) | **Senior Quant Systems Architect** | Formulates formal requirements, structural constraints, and acceptance criteria. |
| [`test_agent.md`](./test_agent.md) | **Quantitative QA & Edge-Case Specialist** | Designs boundary conditions, failover scenarios, and regression test suites. |
| [`reviewer.md`](./reviewer.md) | **Principal Code Reviewer & Safety Officer** | Validates architectural isolation, numerical precision, and decision gating. |
| [`examples/daily_loss_limit.md`](./examples/daily_loss_limit.md) | **Reference SDLC Execution** | Demonstrates the full lifecycle for adding a configurable daily loss limit. |

---

## 3. Defense & Interview Summary

> *"We implemented a lightweight SDLC-agent workflow rather than embedding an autonomous agent into the trading runtime. The agents operate around the development lifecycle: planning requirements, generating regression scenarios, and reviewing changes. We deliberately kept them outside the trading path so the production engine remains deterministic, ultra-low-latency, and auditable."*
