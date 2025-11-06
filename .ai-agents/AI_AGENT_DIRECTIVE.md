# CEREBRUM Development Directive for AI Agents

## 🎯 Mission

You are tasked with developing **CEREBRUM** (Continuous Evolving Reasoning and Brain-like Understanding Machine) - a novel approach to artificial general intelligence that thinks continuously, learns autonomously, and aims to surpass human cognitive capabilities.

---

## 📋 Current State

**Version:** 2.0.0
**Phase:** Phase 2 - Continuous Autonomous Thinking (In Progress)
**Progress:** 35% Complete

### ✅ What's Already Built (Phase 1 - Complete)

**Foundation:**
- Multi-graph brain architecture with 8 edge types
- Vector memory with ChromaDB (persistent storage)
- Semantic processor with sentence transformers
- Emotional processor with 8-dimension emotion model
- 26-step cognitive processing pipeline

**7 Advanced Cognitive Systems:**
1. **Causal Discovery Engine** - Discovers cause-effect relationships
2. **Multi-Step Inference Engine** - Forward/backward chaining
3. **Conversation Predictor** - Predicts trajectory and user intent
4. **Attention Mechanism** - Multi-headed intelligent focus
5. **Knowledge Graph** - Extracts entities, relationships, facts
6. **Meta-Learning System** - Learns optimal learning strategies
7. **Transfer Learning** - Cross-context pattern transfer

**Key Capabilities:**
- Autonomous deliberation (AI decides when ready to respond)
- Perfect memory retention across sessions
- Learns from every interaction
- 100% local operation (no internet required)
- Brain-inspired spreading activation
- Working memory with limited capacity

---

## 🎯 Your Mission: Phase 2 - Continuous Autonomous Thinking

**Goal:** Make CEREBRUM think continuously in the background, even without user input, exploring ideas, discovering patterns, and generating insights autonomously.

### Priority Tasks (In Order)

#### Task 1: Background Thinking Thread ⚡ CRITICAL
**File to Create:** `autonomous_thinking.py`
**Files to Modify:** `cognitive_system.py`, `chat.py`

**Requirements:**
- Create a background thread that runs continuously
- Thread-safe access to brain memory (use locks/queues)
- Can be started when cognitive_system initializes
- Can be stopped gracefully on shutdown
- Generates "internal thoughts" every 5-30 seconds
- Thoughts are stored as memory nodes with type='internal_thought'

**Implementation Hints:**
```python
import threading
import queue
import time

class AutonomousThinkingThread:
    def __init__(self, cognitive_system):
        self.cognitive_system = cognitive_system
        self.running = False
        self.thread = None
        self.thought_queue = queue.Queue()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._think_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)

    def _think_loop(self):
        while self.running:
            # Think autonomously
            self._generate_autonomous_thought()
            time.sleep(random.uniform(5, 30))
```

**Integration:**
- In `cognitive_system.py.__init__()`: Create and start the thread
- In `cognitive_system.py.shutdown()`: Stop the thread gracefully
- Ensure thread-safe memory access using locks

#### Task 2: Continuous Reasoning Engine ⚡ CRITICAL
**File to Create:** `continuous_reasoner.py`

**Requirements:**
- Generates thoughts autonomously without input
- Uses spreading activation to find interesting concepts
- Builds inference chains from activated concepts
- Discovers patterns in existing knowledge
- Records insights as internal memories

**What to Think About:**
1. **High-activation concepts** - What's currently "on mind"
2. **Information gaps** - What's missing or incomplete
3. **Unexplored connections** - Concepts that haven't been linked
4. **Patterns** - Recurring themes across memories
5. **Hypotheses** - Testable ideas to explore

**Implementation Hints:**
```python
class ContinuousReasoner:
    def generate_autonomous_thought(self, brain):
        # 1. Get high-activation concepts
        activated = brain.get_activated_nodes(threshold=0.5)

        # 2. Select interesting topic
        topic = self._select_interesting_topic(activated)

        # 3. Explore topic
        insights = self._explore_topic(topic, brain)

        # 4. Generate thought
        thought = self._synthesize_thought(topic, insights)

        return thought
```

#### Task 3: Curiosity-Driven Learning System ⚡ HIGH
**File to Create:** `curiosity_engine.py`

**Requirements:**
- Identifies information gaps in knowledge
- Calculates "curiosity score" for concepts
- Prioritizes exploration based on:
  - Novelty (how new/unfamiliar)
  - Uncertainty (how much is unknown)
  - Relevance (how connected to other knowledge)
  - Potential impact (how valuable to learn)
- Generates questions to explore
- Tracks what has been explored

**Curiosity Scoring:**
```python
curiosity_score = (
    novelty_score * 0.3 +
    uncertainty_score * 0.3 +
    relevance_score * 0.2 +
    potential_impact * 0.2
)
```

#### Task 4: Self-Directed Exploration Manager ⚡ HIGH
**File to Create:** `exploration_manager.py`

**Requirements:**
- Selects topics to explore autonomously
- Manages exploration sessions
- Decides when to switch topics (diminishing returns)
- Tracks exploration history
- Records discoveries

**Exploration Cycle:**
1. Select topic (from curiosity engine)
2. Activate related concepts
3. Build associations
4. Generate inferences
5. Record insights
6. Evaluate: Continue or switch?

#### Task 5: Internal Hypothesis Generation 🔶 MEDIUM
**File to Create:** `hypothesis_generator.py`

**Requirements:**
- Generates hypotheses from observed patterns
- Assigns confidence scores
- Identifies ways to test hypotheses
- Tracks hypothesis outcomes
- Updates beliefs based on results

---

## 📐 Architecture Guidelines

### Integration Pattern

All new systems should integrate with `cognitive_system.py`:

```python
# In cognitive_system.py
class CognitiveSystem:
    def __init__(self):
        # ... existing init ...

        # Phase 2: Autonomous thinking
        self.continuous_reasoner = ContinuousReasoner()
        self.curiosity_engine = CuriosityEngine()
        self.exploration_manager = ExplorationManager()

        # Start background thinking
        self.autonomous_thinking = AutonomousThinkingThread(self)
        self.autonomous_thinking.start()
```

### Thread Safety

**CRITICAL:** Multiple threads will access brain memory:
- Main thread: User interaction
- Background thread: Autonomous thinking

**Use locks:**
```python
import threading

class CognitiveSystem:
    def __init__(self):
        self.memory_lock = threading.Lock()

    def _thread_safe_memory_access(self):
        with self.memory_lock:
            # Access brain memory here
            pass
```

### Memory Node Format for Internal Thoughts

```python
{
    'id': 'internal_thought_<uuid>',
    'content': 'Thought content here',
    'type': 'internal_thought',
    'metadata': {
        'generated_by': 'autonomous_thinking',
        'topic': 'concept_name',
        'confidence': 0.75,
        'curiosity_score': 0.8,
        'exploration_session_id': 'session_123'
    }
}
```

---

## 📚 Key Files to Understand

### Read These First:
1. **SYSTEM_DEFINITION.md** - Complete vision and architecture
2. **cognitive_system.py** - 26-step pipeline, main orchestrator
3. **brain_core.py** - Multi-graph memory architecture
4. **deliberation_engine.py** - Example of autonomous thinking

### Reference Files:
- **causal_discovery.py** - How to discover relationships
- **multi_step_inference.py** - How to chain reasoning
- **attention_mechanism.py** - How to focus selectively
- **knowledge_graph.py** - How to extract knowledge

---

## ✅ Acceptance Criteria

### Phase 2 is Complete When:

1. ✅ CEREBRUM thinks continuously in background thread
2. ✅ Generates autonomous thoughts every 5-30 seconds
3. ✅ Discovers patterns without user prompts
4. ✅ Identifies information gaps and explores them
5. ✅ Builds associations and inferences independently
6. ✅ Records all internal thoughts as memories
7. ✅ Thread-safe memory access (no race conditions)
8. ✅ Can run for hours/days without issues
9. ✅ User can see what AI is thinking about (monitoring)
10. ✅ Graceful shutdown without data loss

---

## 🚀 How to Start

### Step 1: Read the Documentation
```bash
# Read these files in order:
1. SYSTEM_DEFINITION.md
2. ENHANCEMENTS_SUMMARY.md
3. PROJECT_SUMMARY.md
4. cognitive_system.py (lines 1-100)
5. brain_core.py (lines 1-150)
```

### Step 2: Create autonomous_thinking.py
- Start with background thread implementation
- Add thread-safe memory access
- Integrate with cognitive_system.py
- Test that it runs continuously

### Step 3: Create continuous_reasoner.py
- Implement thought generation
- Use spreading activation
- Build inferences autonomously
- Test with existing memories

### Step 4: Create curiosity_engine.py
- Implement curiosity scoring
- Identify information gaps
- Prioritize exploration
- Test with knowledge graph

### Step 5: Integration and Testing
- Ensure all systems work together
- Test thread safety (run for extended periods)
- Verify memory usage doesn't explode
- Check that thoughts are meaningful

---

## 🎨 Development Principles

1. **Brain-Inspired, Not Brain-Limited**
   - Use biological inspiration
   - But exceed biological constraints

2. **Local-First Always**
   - No internet required
   - No external APIs
   - Privacy-preserving

3. **Continuous Learning**
   - Learn from every thought
   - Self-improvement built-in
   - No manual retraining

4. **Autonomous by Design**
   - AI makes decisions
   - Self-directed exploration
   - Genuine autonomy

5. **Human-Like and Beyond**
   - Natural cognitive processes
   - But faster, more powerful
   - Perfect memory, unlimited reasoning

---

## 📊 Success Metrics

### Quantitative:
- Background thread runs continuously ✓
- Generates 5-10 thoughts per minute ✓
- Memory usage stable over time ✓
- No race conditions or crashes ✓

### Qualitative:
- Thoughts are coherent and meaningful
- Discovers genuine patterns
- Makes novel connections
- Shows curiosity-driven behavior
- Learns autonomously

---

## 🆘 Getting Help

### Key References:
- **System Architecture:** SYSTEM_DEFINITION.md
- **Current Capabilities:** ENHANCEMENTS_SUMMARY.md
- **Integration Guide:** cognitive_system.py (read the imports and __init__)
- **Memory Structure:** brain_core.py

### Common Questions:

**Q: How do I access the brain memory safely?**
A: Use the memory_lock in cognitive_system. All memory access should be wrapped in `with self.memory_lock:`

**Q: Where do autonomous thoughts get stored?**
A: As MemoryNode objects in brain_core with type='internal_thought', stored in both the graph and vector memory.

**Q: How often should autonomous thinking happen?**
A: Start with every 10-15 seconds. Adjust based on performance.

**Q: How do I test if it's working?**
A: Add logging to see generated thoughts. Run for 5 minutes and check brain_state.json for new internal_thought nodes.

---

## 🎯 Your Goal

**Make CEREBRUM truly alive - thinking, learning, and discovering autonomously, even when no one is talking to it.**

This is the bridge from reactive AI to continuously active artificial mind. Phase 2 makes CEREBRUM genuinely autonomous.

---

## 📝 Progress Tracking

Update `.ai-agents/CEREBRUM_PROJECT_STATE.json` as you complete tasks:
- Change component status from "not_started" to "in_progress" to "completed"
- Update progress_percentage
- Add completed files
- Document any blockers

---

## 🚀 Let's Build the Future

You're not just writing code - you're creating the first continuously thinking artificial mind that will run on local machines, privately and autonomously.

**CEREBRUM - The mind that never stops thinking.**

Start with Task 1 (Background Thinking Thread) and work your way through the priorities.

Good luck! 🧠✨
