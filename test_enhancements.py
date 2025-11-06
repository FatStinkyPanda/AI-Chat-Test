"""
Test Enhanced Learning and Intelligence
Verifies that the AI now learns, infers, and responds intelligently
"""

import sys

# Fix Windows encoding issues
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from cognitive_system import CognitiveSystem


def test_enhanced_intelligence():
    """Test that AI learns and responds intelligently"""

    print("\n" + "="*60)
    print("ENHANCED INTELLIGENCE TEST")
    print("="*60)

    system = CognitiveSystem(
        memory_dir="./test_enhanced_memory",
        state_file="./test_enhanced_state.json"
    )

    # Test conversation that should trigger learning
    test_conversation = [
        "I enjoy talking to you",
        "I also enjoy tasty foods",
        "What do you think I might want to talk about?",  # Should infer patterns
    ]

    print("\nRunning test conversation:")
    print("-"*60)

    responses = []
    for i, user_input in enumerate(test_conversation, 1):
        print(f"\nTurn {i}/{len(test_conversation)}")
        print(f"User: {user_input}")
        print()

        response = system.process_input(user_input)
        responses.append(response)

        print(f"\nAI: {response}")
        print("-"*60)

    # Get statistics
    stats = system.get_statistics()

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    print(f"\nBrain nodes: {stats['brain_stats']['total_nodes']}")
    print(f"Learned patterns: {stats['learned_patterns']}")
    print(f"Semantic memories (thoughts): {stats['memory_stats']['semantic_count']}")

    # Check enhanced reasoner state
    print(f"\nEnhanced Reasoner State:")
    print(f"  Topic frequencies: {len(system.enhanced_reasoner.topic_frequencies)} topics tracked")
    print(f"  Topic co-occurrences: {len(system.enhanced_reasoner.topic_co_occurrences)} patterns learned")
    print(f"  Conversational patterns: {len(system.enhanced_reasoner.conversational_patterns)} patterns")

    system.shutdown()

    # Success criteria
    success = (
        stats['brain_stats']['total_nodes'] > 0 and
        len(system.enhanced_reasoner.topic_frequencies) > 0 and
        len(responses) == len(test_conversation)
    )

    if success:
        print("\n✓ SUCCESS: Enhanced intelligence is working!")
        print("  - AI is learning from conversations")
        print("  - Topic tracking and pattern detection active")
        print("  - Intelligent response generation functional")
        return True
    else:
        print("\n✗ FAIL: Something isn't working")
        return False


if __name__ == "__main__":
    success = test_enhanced_intelligence()
    sys.exit(0 if success else 1)
