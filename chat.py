#!/usr/bin/env python3
"""
Brain-Inspired AI Chat Interface
Main entry point for conversational interaction
"""

import sys
import os
from typing import Optional
import argparse
from datetime import datetime

from cognitive_system import CognitiveSystem
from io_interface import IOManager, TextInterface, AudioInterface, VideoInterface, WebInterface


class ChatBot:
    """
    Main chatbot application
    """

    def __init__(self, memory_dir: str = "./brain_memory",
                 state_file: str = "./brain_state.json"):
        """Initialize the chatbot"""
        self.cognitive_system: Optional[CognitiveSystem] = None
        self.io_manager = IOManager()
        self.memory_dir = memory_dir
        self.state_file = state_file
        self.running = False

    def initialize(self):
        """Initialize all components"""
        print("=" * 60)
        print("Brain-Inspired Conversational AI")
        print("A novel multi-graph cognitive architecture")
        print("=" * 60)
        print()

        # Initialize I/O interfaces
        print("Initializing I/O interfaces...")
        text_interface = TextInterface()
        self.io_manager.register_interface(text_interface)

        # Try to register optional interfaces
        audio_interface = AudioInterface()
        self.io_manager.register_interface(audio_interface)

        video_interface = VideoInterface()
        self.io_manager.register_interface(video_interface)

        web_interface = WebInterface()
        self.io_manager.register_interface(web_interface)

        print()

        # Initialize cognitive system
        print("Initializing cognitive system (this may take a moment)...")
        self.cognitive_system = CognitiveSystem(
            memory_dir=self.memory_dir,
            state_file=self.state_file
        )

        print()
        print("=" * 60)
        print("System ready! Type 'help' for commands or start chatting.")
        print("=" * 60)
        print()

    def run(self):
        """Main conversation loop"""
        if not self.cognitive_system:
            self.initialize()

        self.running = True

        # Display welcome message
        welcome = self._get_welcome_message()
        self.io_manager.send_output(welcome)

        # Main loop
        while self.running:
            try:
                # Get user input
                user_input = self.io_manager.receive_input()

                if user_input is None:
                    break

                if not user_input.strip():
                    continue

                # Check for commands
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                    continue

                # Process input through cognitive system
                response = self.cognitive_system.process_input(user_input)

                # Send response
                self.io_manager.send_output(response)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Use /quit to exit properly.")
                continue

            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Shutdown
        self.shutdown()

    def _get_welcome_message(self) -> str:
        """Generate welcome message"""
        stats = self.cognitive_system.get_statistics()

        welcome = "Hello! I'm a brain-inspired AI system that learns and remembers.\n"

        if stats['brain_stats']['total_nodes'] > 0:
            welcome += f"I remember our previous conversations ({stats['brain_stats']['total_nodes']} memories).\n"

        welcome += "How can I help you today?"

        return welcome

    def _handle_command(self, command: str):
        """Handle special commands"""
        cmd = command.lower().strip()

        if cmd == '/quit' or cmd == '/exit':
            print("Goodbye!")
            self.running = False

        elif cmd == '/help':
            self._show_help()

        elif cmd == '/stats':
            self._show_statistics()

        elif cmd == '/reset':
            self.cognitive_system.reset_conversation()
            self.io_manager.send_output("Conversation reset. Starting fresh!")

        elif cmd == '/interfaces':
            self._show_interfaces()

        elif cmd == '/switch':
            self.io_manager.switch_interface()

        elif cmd == '/save':
            self.cognitive_system._save_state()
            print("State saved successfully!")

        elif cmd == '/clear':
            os.system('cls' if os.name == 'nt' else 'clear')

        elif cmd == '/memory':
            self._show_memory_info()

        elif cmd == '/graph':
            self._show_graph_info()

        else:
            print(f"Unknown command: {cmd}")
            print("Type /help for available commands")

    def _show_help(self):
        """Display help information"""
        help_text = """
Available Commands:
  /help       - Show this help message
  /quit       - Exit the program
  /stats      - Show system statistics
  /memory     - Show memory information
  /graph      - Show cognitive graph information
  /reset      - Reset current conversation (keep long-term memory)
  /save       - Manually save current state
  /clear      - Clear the screen
  /interfaces - Show available I/O interfaces
  /switch     - Switch between interfaces

Special Features:
  - Multi-graph memory with semantic, emotional, temporal, and other edge types
  - Persistent memory across sessions (saved in ./brain_memory/)
  - Emotional understanding and empathetic responses
  - Context-aware conversation tracking
  - Pattern learning from interactions
  - Spreading activation across memory network

Architecture:
  - Brain-inspired cognitive architecture
  - Multiple memory systems (episodic, semantic, emotional)
  - Vector database for efficient similarity search
  - Multi-hop reasoning across different edge types
  - Hebbian learning for association strengthening
"""
        print(help_text)

    def _show_statistics(self):
        """Display system statistics"""
        stats = self.cognitive_system.get_statistics()

        print("\n" + "=" * 60)
        print("System Statistics")
        print("=" * 60)

        print(f"\nConversation:")
        print(f"  Turns: {stats['conversation_turns']}")
        print(f"  ID: {stats['conversation_id'][:8]}...")

        print(f"\nBrain (Graph Memory):")
        print(f"  Total nodes: {stats['brain_stats']['total_nodes']}")
        print(f"  Total edges: {stats['brain_stats']['total_edges']}")
        print(f"  Average activation: {stats['brain_stats']['average_activation']:.3f}")

        print(f"\n  Edge types:")
        for edge_type, count in stats['brain_stats']['edge_type_counts'].items():
            print(f"    {edge_type}: {count}")

        print(f"\nVector Memory:")
        print(f"  Episodic memories: {stats['memory_stats']['episodic_count']}")
        print(f"  Semantic memories: {stats['memory_stats']['semantic_count']}")
        print(f"  Emotional memories: {stats['memory_stats']['emotional_count']}")
        print(f"  Total: {stats['memory_stats']['total_memories']}")

        print(f"\nLearning:")
        print(f"  Learned patterns: {stats['learned_patterns']}")

        if stats['brain_stats']['most_accessed_nodes']:
            print(f"\nMost accessed memories:")
            for node_id, count, content in stats['brain_stats']['most_accessed_nodes'][:3]:
                print(f"  [{count}x] {content[:50]}...")

        print("=" * 60 + "\n")

    def _show_memory_info(self):
        """Display memory information"""
        stats = self.cognitive_system.get_statistics()

        print("\n" + "=" * 60)
        print("Memory Information")
        print("=" * 60)

        print(f"\nWorking Memory: {stats['brain_stats']['working_memory_size']} active nodes")
        print("  (Current conversation context)")

        print(f"\nEpisodic Memory: {stats['memory_stats']['episodic_count']} memories")
        print("  (Past conversations and events)")

        print(f"\nSemantic Memory: {stats['memory_stats']['semantic_count']} memories")
        print("  (Facts and knowledge)")

        print(f"\nEmotional Memory: {stats['memory_stats']['emotional_count']} memories")
        print("  (Emotional associations)")

        print("\nMemory is persistent and saved in:")
        print(f"  Vector memory: {self.memory_dir}")
        print(f"  Graph state: {self.state_file}")

        print("=" * 60 + "\n")

    def _show_graph_info(self):
        """Display cognitive graph information"""
        stats = self.cognitive_system.get_statistics()

        print("\n" + "=" * 60)
        print("Cognitive Graph Information")
        print("=" * 60)

        print(f"\nNodes: {stats['brain_stats']['total_nodes']}")
        print(f"Edges: {stats['brain_stats']['total_edges']}")

        if stats['brain_stats']['total_nodes'] > 0:
            density = stats['brain_stats']['total_edges'] / max(stats['brain_stats']['total_nodes'], 1)
            print(f"Graph density: {density:.2f} edges per node")

        print("\nEdge Types (cognitive connections):")
        for edge_type, count in stats['brain_stats']['edge_type_counts'].items():
            print(f"  {edge_type:12s}: {count:4d} connections")

        print("\nEdge Type Descriptions:")
        print("  semantic    : Meaning similarity between concepts")
        print("  emotional   : Shared emotional associations")
        print("  temporal    : Time-based connections")
        print("  causal      : Cause-effect relationships")
        print("  contextual  : Co-occurrence patterns")
        print("  analogical  : Metaphorical connections")
        print("  procedural  : Action-response patterns")
        print("  episodic    : Episodic memory links")

        print("=" * 60 + "\n")

    def _show_interfaces(self):
        """Display available interfaces"""
        capabilities = self.io_manager.get_capabilities()

        print("\n" + "=" * 60)
        print("I/O Interfaces")
        print("=" * 60)

        for name, caps in capabilities.items():
            status = "ACTIVE" if name == self.io_manager.active_interface else "AVAILABLE" if caps['enabled'] else "DISABLED"
            print(f"\n{name.upper()} [{status}]")

            if caps.get('metadata'):
                for key, value in caps['metadata'].items():
                    print(f"  {key}: {value}")

        print("\nUse /switch to change active interface")
        print("=" * 60 + "\n")

    def shutdown(self):
        """Gracefully shutdown the system"""
        print("\nShutting down...")

        if self.cognitive_system:
            self.cognitive_system.shutdown()

        self.io_manager.cleanup_all()

        print("Goodbye!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Brain-Inspired Conversational AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chat.py                    # Start with defaults
  python chat.py --memory ./custom  # Use custom memory directory

Features:
  - Multi-graph cognitive architecture
  - Persistent memory across sessions
  - Emotional understanding
  - Context-aware responses
  - Pattern learning
  - Extensible I/O (text, audio, video, web)
"""
    )

    parser.add_argument(
        '--memory',
        default='./brain_memory',
        help='Directory for persistent memory storage (default: ./brain_memory)'
    )

    parser.add_argument(
        '--state',
        default='./brain_state.json',
        help='File for brain state storage (default: ./brain_state.json)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )

    args = parser.parse_args()

    try:
        # Create chatbot
        chatbot = ChatBot(
            memory_dir=args.memory,
            state_file=args.state
        )

        # Run
        chatbot.run()

    except KeyboardInterrupt:
        print("\n\nInterrupted. Exiting...")
        sys.exit(0)

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
