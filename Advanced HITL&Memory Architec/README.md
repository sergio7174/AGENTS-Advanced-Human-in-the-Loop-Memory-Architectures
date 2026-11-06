# 🤖 Production-Grade AI Agent Architectures (LangGraph & LangChain) - Module 3: Advanced Human-in-the-Loop & Memory Architectures

A comprehensive lab repository demonstrating advanced enterprise AI engineering patterns, including fault-tolerant tool dispatch, durable state persistence, human-in-the-loop (HITL) safety workflows, long-term semantic reflection memory, and multi-agent supervisor routing.

---

## 🏗️ Architectural Overview & Modules

This repository is structured into progressive modules representing real-world production challenges in agentic system design:

### **Module 2: Resilient Tool-Calling & Structured Validation**
* **Strict Schema Enforcement:** Utilizes Pydantic schemas (`.with_structured_output`) to guarantee validated, type-safe agent outputs.
* **Autonomous Error Self-Correction:** Implements robust `try-except` execution boundaries within tools to prevent graph crashes and allow agents to self-correct on bad arguments.

### **Module 3: Durable State Persistence & Human-in-the-Loop (HITL)**
* **PostgreSQL Checkpointing (`PostgresSaver`):** Serializes complete graph state, message history, and pending queues directly to a database, ensuring zero state loss across server reboots.
* **HITL Pause-and-Resume:** Leverages declarative compilation breakpoints (`interrupt_before=["tools"]`) and manual state overrides (`graph.update_state`) to require human sign-off before executing critical actions.
* **Semantic Reflection Memory:** Implements background summarization loops combined with `RemoveMessage` pruning to eliminate token bloat while maintaining long-term context.
* **Multi-Tenant Data Isolation:** Enforces composite thread namespacing (`user_id + session_id`) and partitioned table schemas to prevent cross-session and cross-user data leaks.

### **Module 4: Multi-Agent Collaboration & Supervisor Routing**
* **Hierarchical Supervisor-Worker Pattern:** Employs a central router LLM equipped with structured output decisions to dynamically delegate sub-tasks to specialized worker agents (e.g., researcher, coder).
* **Modular Execution Control:** Keeps individual agent system prompts focused and decoupled, preventing context pollution and reasoning degradation.

---

## 🛠️ Tech Stack & Prerequisites

* **Language:** Python 3.13+
* **Orchestration:** LangGraph, LangChain
* **LLM Providers:** Google Gemini API (`gemini-2.5-flash`), OpenRouter (Llama 3.3)
* **State Persistence & Database:** PostgreSQL (`psycopg`, `PostgresSaver`)
* **Validation:** Pydantic v2

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/ai-agents-lab.git](https://github.com/your-username/ai-agents-lab.git)
cd ai-agents-lab