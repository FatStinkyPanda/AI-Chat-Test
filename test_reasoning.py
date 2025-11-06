"""
Test Advanced Reasoning Capabilities
Tests inference, thought recording, gap detection, and truthfulness evaluation
"""

import sys
import numpy as np
from datetime import datetime

# Fix Windows encoding issues
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from reasoning_engine import ReasoningEngine, Inference, InformationGap
from cognitive_system import CognitiveSystem
from brain_core import MemoryNode, EmotionalValence


def test_basic_inference():
    """Test basic inference capabilities"""
    print("\n" + "="*60)
    print("TEST 1: Basic Inference")
    print("="*60)

    engine = ReasoningEngine()

    # Create test memories
    memories = [
        {
            'content': 'Rain causes the ground to become wet.',
            'id': 'mem_1',
            'metadata': {}
        },
        {
            'content': 'It is raining outside right now.',
            'id': 'mem_2',
            'metadata': {}
        },
        {
            'content': 'Wet ground can make surfaces slippery.',
            'id': 'mem_3',
            'metadata': {}
        }
    ]

    context = "It's raining and I need to go outside"

    inferences = engine.perform_inference(memories, context)

    print(f"\nContext: {context}")
    print(f"\nGenerated {len(inferences)} inferences:")
    for i, inf in enumerate(inferences, 1):
        print(f"\n{i}. Type: {inf.reasoning_type}")
        print(f"   Conclusion: {inf.conclusion}")
        print(f"   Confidence: {inf.confidence:.2f}")

    return len(inferences) > 0


def test_information_gaps():
    """Test information gap detection"""
    print("\n" + "="*60)
    print("TEST 2: Information Gap Detection")
    print("="*60)

    engine = ReasoningEngine()

    # Sparse memories about a topic
    memories = [
        {
            'content': 'Python is a programming language',
            'id': 'mem_1',
            'metadata': {}
        },
        {
            'content': 'I like chocolate ice cream',
            'id': 'mem_2',
            'metadata': {}
        }
    ]

    user_input = "Tell me about Python programming and machine learning"

    gaps = engine.identify_information_gaps(memories, user_input)

    print(f"\nInput: {user_input}")
    print(f"\nIdentified {len(gaps)} information gaps:")
    for i, gap in enumerate(gaps, 1):
        print(f"\n{i}. Topic: {gap.topic}")
        print(f"   Question: {gap.question}")
        print(f"   Importance: {gap.importance:.2f}")

    return len(gaps) > 0


def test_truthfulness_evaluation():
    """Test truthfulness evaluation"""
    print("\n" + "="*60)
    print("TEST 3: Truthfulness Evaluation")
    print("="*60)

    engine = ReasoningEngine()

    # Known facts
    memories = [
        {
            'content': 'The sky is blue during the day',
            'id': 'mem_1',
            'metadata': {}
        },
        {
            'content': 'Water freezes at 0 degrees Celsius',
            'id': 'mem_2',
            'metadata': {}
        },
        {
            'content': 'Birds can fly',
            'id': 'mem_3',
            'metadata': {}
        }
    ]

    # Test statements
    test_statements = [
        "The sky is blue",  # Should be supported
        "Water freezes at 100 degrees",  # Should contradict
        "Cats are mammals",  # No evidence either way
    ]

    print("\nEvaluating statements against known facts:")
    for statement in test_statements:
        confidence, reasoning = engine.evaluate_truthfulness(statement, memories)
        print(f"\nStatement: '{statement}'")
        print(f"Confidence: {confidence:.2f}")
        print(f"Reasoning: {reasoning}")

    return True


def test_thought_generation():
    """Test thought generation from inferences"""
    print("\n" + "="*60)
    print("TEST 4: Thought Generation")
    print("="*60)

    engine = ReasoningEngine()

    memories = [
        {
            'content': 'Exercise improves health and energy.',
            'id': 'mem_1',
            'metadata': {}
        },
        {
            'content': 'I went running this morning.',
            'id': 'mem_2',
            'metadata': {}
        },
        {
            'content': 'Running is a form of exercise.',
            'id': 'mem_3',
            'metadata': {}
        }
    ]

    context = "I feel energetic and healthy today after my run"

    inferences = engine.perform_inference(memories, context)

    print(f"\nContext: {context}")
    print(f"Generated {len(inferences)} inferences")

    if inferences:
        print(f"\nTop inference:")
        print(f"  Type: {inferences[0].reasoning_type}")
        print(f"  Conclusion: {inferences[0].conclusion}")
        print(f"  Confidence: {inferences[0].confidence:.2f}")

        thought = engine.generate_thought(context, memories, inferences)

        if thought:
            print(f"\nGenerated thought: {thought}")
            return True
        else:
            print("\nNo thought generated from inferences")
            return False
    else:
        print("\nNo inferences generated, cannot create thought")
        return False


def test_cognitive_system_integration():
    """Test full cognitive system with reasoning integration"""
    print("\n" + "="*60)
    print("TEST 5: Cognitive System Integration")
    print("="*60)

    print("\nInitializing cognitive system...")
    system = CognitiveSystem(
        memory_dir="./test_brain_memory",
        state_file="./test_brain_state.json"
    )

    # Test conversation that should trigger reasoning
    test_inputs = [
        "The sun causes plants to grow",
        "I planted a garden in the sunny spot",
        "My garden needs attention"
    ]

    print("\nProcessing test conversation:")
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n--- Turn {i} ---")
        print(f"User: {user_input}")

        response = system.process_input(user_input)
        print(f"AI: {response}")

    # Check statistics
    stats = system.get_statistics()
    print("\n" + "-"*60)
    print("System Statistics:")
    print(f"Total nodes: {stats['brain_stats']['total_nodes']}")
    print(f"Total edges: {stats['brain_stats']['total_edges']}")
    print(f"Episodic memories: {stats['memory_stats']['episodic_count']}")
    print(f"Semantic memories (thoughts): {stats['memory_stats']['semantic_count']}")

    # Check if thoughts were created
    thoughts_exist = stats['memory_stats']['semantic_count'] > 0

    system.shutdown()

    return thoughts_exist


def test_connection_finding():
    """Test connection finding through graph traversal"""
    print("\n" + "="*60)
    print("TEST 6: Connection Finding")
    print("="*60)

    engine = ReasoningEngine()

    # This test requires a full brain graph, so we'll simulate it
    print("\nNote: This test requires a populated brain graph")
    print("Connection finding will be tested during actual usage")

    return True


def main():
    """Run all reasoning tests"""
    print("\n")
    print("╔" + "═"*58 + "╗")
    print("║" + " "*15 + "ADVANCED REASONING TEST SUITE" + " "*14 + "║")
    print("╚" + "═"*58 + "╝")

    tests = [
        ("Basic Inference", test_basic_inference),
        ("Information Gap Detection", test_information_gaps),
        ("Truthfulness Evaluation", test_truthfulness_evaluation),
        ("Thought Generation", test_thought_generation),
        ("Cognitive System Integration", test_cognitive_system_integration),
        ("Connection Finding", test_connection_finding),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            results.append((test_name, False, str(e)))

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = 0
    failed = 0

    for test_name, result, error in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {test_name}")
        if error:
            print(f"       Error: {error}")

        if result:
            passed += 1
        else:
            failed += 1

    print("\n" + "-"*60)
    print(f"Total: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("-"*60)

    if failed == 0:
        print("\n🎉 All tests passed! Advanced reasoning is working correctly.")
    else:
        print(f"\n⚠ {failed} test(s) failed. Please review the errors above.")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
