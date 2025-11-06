"""
Modular I/O Interface System
Extensible input/output handling for various modalities
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from datetime import datetime
import json


class IOInterface(ABC):
    """
    Abstract base class for I/O interfaces
    Allows easy extension to audio, video, or other modalities
    """

    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def receive_input(self) -> Optional[str]:
        """Receive input from this interface"""
        pass

    @abstractmethod
    def send_output(self, content: str) -> bool:
        """Send output through this interface"""
        pass

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the interface"""
        pass

    @abstractmethod
    def cleanup(self) -> bool:
        """Cleanup resources"""
        pass

    def is_available(self) -> bool:
        """Check if interface is available"""
        return self.enabled

    def get_capabilities(self) -> Dict[str, Any]:
        """Get interface capabilities"""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'input': True,
            'output': True,
            'metadata': self.metadata
        }


class TextInterface(IOInterface):
    """
    Text-based input/output interface (console/terminal)
    """

    def __init__(self, prompt: str = "You: ", bot_prefix: str = "AI: "):
        super().__init__("text")
        self.prompt = prompt
        self.bot_prefix = bot_prefix
        self.history: List[Dict[str, Any]] = []

    def initialize(self) -> bool:
        """Initialize text interface"""
        print("Text interface initialized")
        return True

    def receive_input(self) -> Optional[str]:
        """Receive text input from user"""
        try:
            user_input = input(self.prompt).strip()

            if user_input:
                self.history.append({
                    'type': 'input',
                    'content': user_input,
                    'timestamp': datetime.now().isoformat()
                })

            return user_input if user_input else None

        except (EOFError, KeyboardInterrupt):
            return None

    def send_output(self, content: str) -> bool:
        """Display text output"""
        try:
            print(f"{self.bot_prefix}{content}")

            self.history.append({
                'type': 'output',
                'content': content,
                'timestamp': datetime.now().isoformat()
            })

            return True

        except Exception as e:
            print(f"Error sending output: {e}")
            return False

    def cleanup(self) -> bool:
        """Cleanup text interface"""
        return True

    def save_history(self, filepath: str):
        """Save conversation history"""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load_history(self, filepath: str):
        """Load conversation history"""
        try:
            with open(filepath, 'r') as f:
                self.history = json.load(f)
        except Exception as e:
            print(f"Could not load history: {e}")


class AudioInterface(IOInterface):
    """
    Audio input/output interface (placeholder for future implementation)
    This demonstrates the extensibility of the system
    """

    def __init__(self):
        super().__init__("audio")
        self.enabled = False  # Not implemented yet
        self.metadata = {
            'sample_rate': 16000,
            'channels': 1,
            'format': 'wav'
        }

    def initialize(self) -> bool:
        """Initialize audio interface"""
        print("Audio interface not yet implemented")
        print("To add audio support:")
        print("  1. Install: pip install speech_recognition pyaudio pyttsx3")
        print("  2. Implement speech-to-text and text-to-speech")
        print("  3. Set self.enabled = True")
        return False

    def receive_input(self) -> Optional[str]:
        """Receive audio input and convert to text"""
        # Future implementation:
        # 1. Capture audio from microphone
        # 2. Use speech_recognition to convert to text
        # 3. Return text
        raise NotImplementedError("Audio input not yet implemented")

    def send_output(self, content: str) -> bool:
        """Convert text to speech and play"""
        # Future implementation:
        # 1. Use pyttsx3 or gTTS to convert text to speech
        # 2. Play audio output
        raise NotImplementedError("Audio output not yet implemented")

    def cleanup(self) -> bool:
        """Cleanup audio resources"""
        return True


class VideoInterface(IOInterface):
    """
    Video input/output interface (placeholder for future implementation)
    Could handle webcam input, screen capture, or video generation
    """

    def __init__(self):
        super().__init__("video")
        self.enabled = False  # Not implemented yet
        self.metadata = {
            'resolution': '640x480',
            'fps': 30,
            'format': 'mp4'
        }

    def initialize(self) -> bool:
        """Initialize video interface"""
        print("Video interface not yet implemented")
        print("To add video support:")
        print("  1. Install: pip install opencv-python")
        print("  2. Implement video capture and processing")
        print("  3. Integrate with vision models for understanding")
        return False

    def receive_input(self) -> Optional[str]:
        """Receive video input and extract information"""
        # Future implementation:
        # 1. Capture frame from webcam
        # 2. Process with vision model
        # 3. Convert to text description
        raise NotImplementedError("Video input not yet implemented")

    def send_output(self, content: str) -> bool:
        """Display or generate video output"""
        # Future implementation:
        # 1. Generate visualization
        # 2. Display on screen or save to file
        raise NotImplementedError("Video output not yet implemented")

    def cleanup(self) -> bool:
        """Cleanup video resources"""
        return True


class WebInterface(IOInterface):
    """
    Web-based interface (placeholder for future implementation)
    Could serve as REST API or WebSocket server
    """

    def __init__(self, host: str = "localhost", port: int = 8000):
        super().__init__("web")
        self.host = host
        self.port = port
        self.enabled = False  # Not implemented yet
        self.metadata = {
            'host': host,
            'port': port,
            'protocol': 'websocket'
        }

    def initialize(self) -> bool:
        """Initialize web interface"""
        print("Web interface not yet implemented")
        print("To add web support:")
        print("  1. Install: pip install fastapi uvicorn websockets")
        print("  2. Implement REST API or WebSocket server")
        print("  3. Create web frontend")
        return False

    def receive_input(self) -> Optional[str]:
        """Receive input from web client"""
        raise NotImplementedError("Web interface not yet implemented")

    def send_output(self, content: str) -> bool:
        """Send output to web client"""
        raise NotImplementedError("Web interface not yet implemented")

    def cleanup(self) -> bool:
        """Cleanup web server"""
        return True


class IOManager:
    """
    Manages multiple I/O interfaces
    Allows simultaneous use of multiple input/output modalities
    """

    def __init__(self):
        self.interfaces: Dict[str, IOInterface] = {}
        self.active_interface: Optional[str] = None

    def register_interface(self, interface: IOInterface) -> bool:
        """Register a new I/O interface"""
        if interface.initialize():
            self.interfaces[interface.name] = interface
            if self.active_interface is None:
                self.active_interface = interface.name
            print(f"Registered interface: {interface.name}")
            return True
        else:
            print(f"Failed to initialize interface: {interface.name}")
            return False

    def set_active_interface(self, name: str) -> bool:
        """Set the active interface"""
        if name in self.interfaces and self.interfaces[name].is_available():
            self.active_interface = name
            return True
        return False

    def receive_input(self, interface_name: Optional[str] = None) -> Optional[str]:
        """Receive input from specified or active interface"""
        interface_name = interface_name or self.active_interface

        if interface_name and interface_name in self.interfaces:
            return self.interfaces[interface_name].receive_input()

        return None

    def send_output(self, content: str, interface_name: Optional[str] = None) -> bool:
        """Send output through specified or active interface"""
        interface_name = interface_name or self.active_interface

        if interface_name and interface_name in self.interfaces:
            return self.interfaces[interface_name].send_output(content)

        return False

    def broadcast_output(self, content: str) -> Dict[str, bool]:
        """Send output through all available interfaces"""
        results = {}
        for name, interface in self.interfaces.items():
            if interface.is_available():
                results[name] = interface.send_output(content)
        return results

    def get_available_interfaces(self) -> List[str]:
        """Get list of available interfaces"""
        return [
            name for name, interface in self.interfaces.items()
            if interface.is_available()
        ]

    def get_capabilities(self) -> Dict[str, Dict]:
        """Get capabilities of all interfaces"""
        return {
            name: interface.get_capabilities()
            for name, interface in self.interfaces.items()
        }

    def cleanup_all(self):
        """Cleanup all interfaces"""
        for interface in self.interfaces.values():
            interface.cleanup()

    def switch_interface(self):
        """Interactive interface switching"""
        available = self.get_available_interfaces()

        if len(available) <= 1:
            print("No other interfaces available to switch to")
            return

        print("\nAvailable interfaces:")
        for i, name in enumerate(available, 1):
            active_marker = " (active)" if name == self.active_interface else ""
            print(f"  {i}. {name}{active_marker}")

        try:
            choice = input("\nSelect interface (number): ").strip()
            idx = int(choice) - 1

            if 0 <= idx < len(available):
                self.set_active_interface(available[idx])
                print(f"Switched to {available[idx]} interface")
            else:
                print("Invalid selection")

        except (ValueError, KeyboardInterrupt):
            print("Cancelled")
