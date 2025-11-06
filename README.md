# Brain-Inspired Conversational AI

A novel multi-graph cognitive architecture for natural language understanding and generation with persistent memory.

## Quick Start

**Don't have Python 3.10? No problem!**

Just run this and follow the prompts:
```bash
python setup.py
```

The setup will automatically detect your Python version and offer to install Python 3.10.11 if needed. It will ask for your permission before making any changes.

---

## Requirements

**Python Version: 3.10.x** (Required for full compatibility with ChromaDB and numpy)

- **Recommended**: Python 3.10.11
- **Auto-installer available**: Run `python install_python.py` for guided installation
- **Not compatible**: Python 3.11+ or Python 3.9 and below may have dependency issues

## Overview

This project implements a brain-inspired AI system that:
- **Learns from conversations** through multi-graph memory architecture
- **Remembers everything** using persistent vector and graph-based storage
- **Understands emotions** with affective processing
- **Reasons contextually** using multi-hop traversal across different edge types
- **Supports extensible I/O** for text, audio, video, and web interfaces

## Architecture

### Core Components

1. **Multi-Graph Memory System** (`brain_core.py`)
   - Multiple edge types: semantic, emotional, temporal, causal, contextual, analogical, procedural, episodic
   - Spreading activation for memory recall
   - Hebbian learning for connection strengthening
   - Working memory management

2. **Semantic Processing** (`semantic_processor.py`)
   - Sentence transformer embeddings
   - Semantic similarity and analogy detection
   - Context aggregation and management
   - Keyword extraction

3. **Emotional Processing** (`emotional_processor.py`)
   - Multi-dimensional emotion analysis
   - Sentiment detection
   - Emotional congruence checking
   - Empathetic response generation

4. **Vector Memory** (`vector_memory.py`)
   - ChromaDB for persistent storage
   - Episodic, semantic, and emotional memory collections
   - Hybrid vector + keyword search
   - Metadata filtering

5. **Cognitive System** (`cognitive_system.py`)
   - Unified cognitive pipeline
   - Pattern learning from interactions
   - Multi-hop reasoning
   - Response generation

6. **I/O Interface System** (`io_interface.py`)
   - Modular, extensible I/O
   - Text interface (implemented)
   - Audio, video, web interfaces (extensible)

## Key Features

### Brain-Like Architecture
- **Multiple Edge Types**: Like neural connections, memories are linked through different relationship types
- **Spreading Activation**: Thinking about one concept activates related memories
- **Working Memory**: Limited capacity short-term context
- **Long-Term Memory**: Unlimited persistent storage across sessions
- **Hebbian Learning**: "Neurons that fire together, wire together"

### Memory Systems
- **Episodic Memory**: Past conversations and events
- **Semantic Memory**: Facts and knowledge
- **Emotional Memory**: Affective associations
- **Procedural Memory**: Learned patterns

### Advanced Reasoning
- Multi-hop traversal across different edge types
- Context-aware retrieval
- Analogical reasoning
- Temporal and causal understanding

## Installation

### Automatic Installation (Recommended)

The project includes an automatic Python version manager that will install Python 3.10.x for you if needed.

**Simply run:**
```bash
python setup.py
```

The setup will:
1. Check your Python version
2. If incompatible, offer to install Python 3.10.11 automatically
3. Ask for your permission before installing
4. Guide you through the installation process
5. Install all required dependencies

### Manual Installation

If you prefer to install Python manually:

#### Prerequisites
- **Python 3.10.x** (download from [python.org](https://www.python.org/downloads/release/python-31011/))
- pip (included with Python)

**Check your Python version:**
```bash
python --version
```

#### Setup Steps

1. **Clone the repository:**
```bash
git clone https://github.com/FatStinkyPanda/AI-Chat-Test.git
cd AI-Chat-Test
```

2. **Automatic Python installation (if needed):**
```bash
python install_python.py
```
This will detect your Python version and offer to install 3.10.11 if incompatible.

3. **Install dependencies:**
```bash
python setup.py
```

4. **Run the AI:**
```bash
python chat.py
```

### Troubleshooting Installation

**If you have Python version issues:**
```bash
# Check current version
python --version

# Run automatic installer
python install_python.py

# Or manually verify
python check_python.py
```

**If dependencies fail to install:**
```bash
# Try manual installation
pip install -r requirements.txt

# For ChromaDB issues specifically
python install_chromadb.py
```

## Usage

### Basic Chat

Start the conversational AI:
```bash
python chat.py
```

### Commands

Once running, you can use these commands:

- `/help` - Show help information
- `/stats` - Display system statistics
- `/memory` - Show memory information
- `/graph` - Show cognitive graph details
- `/reset` - Start new conversation (keeps long-term memory)
- `/save` - Manually save state
- `/quit` - Exit the program

### Command-Line Options

```bash
python chat.py --memory ./custom_memory --state ./custom_state.json
```

Options:
- `--memory DIR` - Custom memory directory (default: ./brain_memory)
- `--state FILE` - Custom state file (default: ./brain_state.json)
- `--debug` - Enable debug mode

## Architecture Details

### Cognitive Pipeline

1. **Perception**: Encode input (text → embeddings)
2. **Memory Creation**: Create new memory node
3. **Storage**: Store in vector database
4. **Context Update**: Update conversation context
5. **Retrieval**: Find relevant memories
6. **Association**: Create cognitive edges
7. **Activation**: Spread activation in network
8. **Reasoning**: Multi-hop traversal
9. **Learning**: Update patterns
10. **Generation**: Generate response
11. **Consolidation**: Memory decay and working memory update

### Edge Types

| Edge Type | Description | Example |
|-----------|-------------|---------|
| Semantic | Meaning similarity | "dog" ↔ "puppy" |
| Emotional | Shared emotions | Happy memories linked |
| Temporal | Time-based | Sequential events |
| Causal | Cause-effect | "rain" → "wet" |
| Contextual | Co-occurrence | Words appearing together |
| Analogical | Metaphorical | "time is money" |
| Procedural | Action-response | Learned behaviors |
| Episodic | Memory episodes | Conversation turns |

## Extending the System

### Adding Audio Support

1. Uncomment audio dependencies in `requirements.txt`
2. Install: `pip install SpeechRecognition pyaudio pyttsx3`
3. Implement speech-to-text in `AudioInterface.receive_input()`
4. Implement text-to-speech in `AudioInterface.send_output()`
5. Set `self.enabled = True`

### Adding Video Support

1. Install: `pip install opencv-python`
2. Implement video capture in `VideoInterface.receive_input()`
3. Integrate with vision model for understanding
4. Set `self.enabled = True`

### Adding Web Interface

1. Install: `pip install fastapi uvicorn websockets`
2. Implement REST API or WebSocket server
3. Create web frontend
4. Set `self.enabled = True`

## Memory Persistence

All memory is automatically saved:
- **Vector memory**: `./brain_memory/` directory (ChromaDB)
- **Graph state**: `./brain_state.json` file

Memory persists across sessions, allowing the AI to remember all previous conversations.

## System Statistics

View statistics with `/stats` command:
- Total memories stored
- Edge type distribution
- Most accessed memories
- Activation levels
- Learned patterns

## Technical Details

### Dependencies
- `sentence-transformers`: Semantic embeddings
- `chromadb`: Vector database
- `torch`: Deep learning backend
- `numpy`: Numerical computing
- `scikit-learn`: ML utilities

### Performance
- First run downloads sentence transformer model (~90MB)
- Subsequent runs load from cache
- Memory usage scales with conversation length
- Automatic state saving every 10 turns

## Project Structure

```
.
├── brain_core.py           # Multi-graph memory system
├── semantic_processor.py   # Semantic understanding
├── emotional_processor.py  # Emotional processing
├── vector_memory.py        # Vector database interface
├── cognitive_system.py     # Unified cognitive system
├── io_interface.py         # I/O interface system
├── chat.py                 # Main chat application
├── requirements.txt        # Python dependencies
├── setup.py               # Setup script
├── README.md              # This file
├── brain_memory/          # Vector memory storage (created on first run)
└── brain_state.json       # Graph memory state (created on first run)
```

## Design Philosophy

This system is designed around principles of human cognition:

1. **Multiple Memory Systems**: Like humans, separate episodic, semantic, and emotional memory
2. **Associative Memory**: Memories linked through meaningful relationships
3. **Context Sensitivity**: Understanding depends on context
4. **Emotional Intelligence**: Recognizing and responding to emotions
5. **Learning Through Interaction**: Improves with use
6. **Persistent Memory**: Never forgets

## Future Enhancements

Potential improvements:
- [ ] More sophisticated response generation (integrate with LLMs)
- [ ] Audio/video I/O implementation
- [ ] Web interface
- [ ] Memory consolidation (sleep-like process)
- [ ] Attention mechanisms
- [ ] Multi-agent conversations
- [ ] Knowledge graph integration
- [ ] Transfer learning capabilities

## License

This is a novel research implementation demonstrating brain-inspired AI architecture.

## Author

Created as a demonstration of multi-graph cognitive architecture for conversational AI.

---

**Note**: This is a novel architecture designed to demonstrate brain-like cognitive processes in AI. The system learns and improves through interaction while maintaining complete memory persistence.
