"""
Test Deliberation and Autonomous Thinking
Tests the AI's ability to think for itself and decide when ready to respond
"""

import sys

# Fix Windows encoding issues
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from cognitive_system import CognitiveSystem


def test_deliberation():
    """Test the deliberation engine with real conversation"""

    print("\n" + "="*60)
    print("DELIBERATION AND AUTONOMOUS THINKING TEST")
    print("="*60)
    print("\nInitializing cognitive system...")

    system = CognitiveSystem(
        memory_dir="./test_deliberation_memory",
        state_file="./test_deliberation_state.json"
    )

    # Test conversations that should trigger different amounts of thinking
    test_conversations = [
        {
            'input': "If it rains, the ground gets wet. It's raining now.",
            'expected_behavior': "Should make causal inference"
        },
        {
            'input': "What do you think about machine learning and artificial intelligence?",
            'expected_behavior': "Should identify information gaps and ask questions"
        },
        {
            'input': "I'm feeling really happy today because I accomplished something important!",
            'expected_behavior': "Should respond with emotional awareness"
        },
        {
            'input': "Can you explain how photosynthesis works in plants?",
            'expected_behavior': "Should think about what to say given limited knowledge"
        },
    ]

    print(f"\nTesting {len(test_conversations)} different conversation scenarios\n")

    for i, test in enumerate(test_conversations, 1):
        print("\n" + "="*60)
        print(f"TEST {i}/{len(test_conversations)}")
        print("="*60)
        print(f"\nExpected Behavior: {test['expected_behavior']}")
        print(f"\nUser: {test['input']}")
        print("-"*60)

        # Process input - this will show deliberation process
        response = system.process_input(test['input'])

        print(f"\nAI Response: {response}")
        print("-"*60)

        input("\nPress Enter to continue to next test...")

    # Get statistics
    stats = system.get_statistics()

    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    print(f"\nTotal nodes in brain: {stats['brain_stats']['total_nodes']}")
    print(f"Total edges: {stats['brain_stats']['total_edges']}")
    print(f"Episodic memories: {stats['memory_stats']['episodic_count']}")
    print(f"Semantic memories (thoughts): {stats['memory_stats']['semantic_count']}")
    print(f"Learned patterns: {stats['learned_patterns']}")

    system.shutdown()

    print("\n" + "="*60)
    print("DELIBERATION TEST COMPLETE")
    print("="*60)
    print("\nKey Observations:")
    print("1. AI thinks iteratively before responding")
    print("2. AI decides autonomously when ready to respond")
    print("3. Stopping reasons vary based on confidence and insights")
    print("4. Deliberation thoughts are saved as memory nodes")
    print("5. AI can learn from its own thinking process")
    print("\nThe AI is now thinking for itself!")


if __name__ == "__main__":
    test_deliberation()
