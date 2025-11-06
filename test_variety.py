"""
Test Response Variety and Naturalness
Verifies that responses are varied and not repetitive
"""

import sys

# Fix Windows encoding issues
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from cognitive_system import CognitiveSystem


def test_response_variety():
    """Test that responses are varied and natural"""

    print("\n" + "="*60)
    print("RESPONSE VARIETY TEST")
    print("="*60)

    system = CognitiveSystem(
        memory_dir="./test_variety_memory",
        state_file="./test_variety_state.json"
    )

    # Test conversation
    conversation = [
        "Hello",
        "What do you enjoy?",
        "I'm interested in the mind and consciousness.",
        "Of course! The mind is a fantastic thing. It can be very powerful.",
        "What do you think about creativity?",
    ]

    print("\nRunning conversation to test variety:")
    print("-"*60)

    responses = []
    for i, user_input in enumerate(conversation, 1):
        print(f"\nTurn {i}/{len(conversation)}")
        print(f"User: {user_input}")

        response = system.process_input(user_input)
        responses.append(response)

        print(f"AI: {response}")
        print("-"*60)

    # Check for variety
    print("\n" + "="*60)
    print("VARIETY ANALYSIS")
    print("="*60)

    # Check if responses start with different phrases
    unique_starts = set()
    for response in responses:
        # Get first 4 words
        words = response.split()[:4]
        start = " ".join(words)
        unique_starts.add(start)

    variety_score = len(unique_starts) / len(responses)

    print(f"\nResponse count: {len(responses)}")
    print(f"Unique response starts: {len(unique_starts)}")
    print(f"Variety score: {variety_score:.2f} (higher is better)")

    # Check for repetitive patterns
    repetitive_phrases = [
        "This relates to when you mentioned",
        "I'm connecting these ideas",
    ]

    repetition_count = 0
    for response in responses:
        for phrase in repetitive_phrases:
            if phrase in response:
                repetition_count += 1

    print(f"Repetitive phrases found: {repetition_count}")

    system.shutdown()

    # Success if variety is good
    success = variety_score > 0.6 and repetition_count < 3

    if success:
        print("\n✓ SUCCESS: Responses are varied and natural!")
        print("  - Good variety in response structure")
        print("  - Limited repetitive patterns")
        return True
    else:
        print("\n⚠ NEEDS IMPROVEMENT:")
        if variety_score <= 0.6:
            print(f"  - Low variety score: {variety_score:.2f}")
        if repetition_count >= 3:
            print(f"  - Too many repetitive phrases: {repetition_count}")
        return False


if __name__ == "__main__":
    success = test_response_variety()
    sys.exit(0 if success else 1)
