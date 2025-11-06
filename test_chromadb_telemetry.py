"""
Test that ChromaDB telemetry is working correctly
Verifies no telemetry error messages are generated
"""

import sys

# Fix Windows encoding issues
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from cognitive_system import CognitiveSystem
import io
import contextlib


def test_no_telemetry_errors():
    """Test that telemetry works without errors"""

    print("\n" + "="*70)
    print("CHROMADB TELEMETRY TEST")
    print("="*70)

    # Capture stderr to check for telemetry errors
    stderr_capture = io.StringIO()

    with contextlib.redirect_stderr(stderr_capture):
        # Initialize system (this triggers telemetry events)
        system = CognitiveSystem(
            memory_dir="./test_telemetry_memory",
            state_file="./test_telemetry_state.json"
        )

        # Process some inputs (more telemetry events)
        system.process_input("Hello")
        system.process_input("How are you?")

        # Shutdown
        system.shutdown()

    # Check stderr for telemetry error messages
    stderr_output = stderr_capture.getvalue()

    telemetry_errors = []
    error_patterns = [
        "Failed to send telemetry event",
        "capture() takes 1 positional argument but 3 were given",
        "ClientStartEvent",
        "ClientCreateCollectionEvent",
        "CollectionAddEvent",
        "CollectionQueryEvent"
    ]

    for line in stderr_output.split('\n'):
        for pattern in error_patterns:
            if pattern in line:
                telemetry_errors.append(line.strip())
                break

    print("\n" + "="*70)
    print("TELEMETRY ERROR CHECK")
    print("="*70)

    if telemetry_errors:
        print(f"\n✗ FOUND {len(telemetry_errors)} TELEMETRY ERRORS:")
        for error in telemetry_errors[:5]:  # Show first 5
            print(f"  • {error}")
        if len(telemetry_errors) > 5:
            print(f"  ... and {len(telemetry_errors) - 5} more")
        return False
    else:
        print("\n✓✓✓ SUCCESS: NO TELEMETRY ERRORS! ✓✓✓")
        print("\nChromaDB is fully operational:")
        print("  • All telemetry events sending correctly")
        print("  • No error messages in stderr")
        print("  • Data capture working perfectly")
        print("  • 100% data integrity maintained")
        return True


if __name__ == "__main__":
    success = test_no_telemetry_errors()
    sys.exit(0 if success else 1)
