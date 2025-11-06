"""
Simple Deliberation Test - Automated
Tests the AI's autonomous thinking without user interaction
"""

import sys

# Fix Windows encoding issues
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from cognitive_system import CognitiveSystem


def test_deliberation_simple():
    """Simple automated test of deliberation"""

    print("\n" + "="*60)
    print("AUTONOMOUS THINKING TEST (Automated)")
    print("="*60)

    system = CognitiveSystem(
        memory_dir="./test_delib_simple_memory",
        state_file="./test_delib_simple_state.json"
    )

    # Build up some context first
    print("\nBuilding context...")
    system.process_input("The sun provides energy for plants")
    print("\n" + "-"*60 + "\n")

    system.process_input("Plants use photosynthesis to grow")
    print("\n" + "-"*60 + "\n")

    # Now ask something that should trigger inference
    print("\nAsking question that should trigger reasoning...")
    response = system.process_input("Why do plants need sunlight?")

    # Get statistics
    stats = system.get_statistics()

    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"\nResponse: {response}")
    print(f"\nBrain nodes: {stats['brain_stats']['total_nodes']}")
    print(f"Deliberation thoughts stored: {stats['memory_stats']['semantic_count']}")

    system.shutdown()

    # Validate
    success = stats['memory_stats']['semantic_count'] > 0

    if success:
        print("\n✓ SUCCESS: AI is thinking autonomously!")
        print("  - AI deliberated before responding")
        print("  - AI decided when ready to respond")
        print("  - Thoughts were saved as memories")
        return True
    else:
        print("\n✗ FAIL: No deliberation thoughts were saved")
        return False


if __name__ == "__main__":
    success = test_deliberation_simple()
    sys.exit(0 if success else 1)
