"""
Event Recorder - Mouse and Keyboard Activity Tracker

DESCRIPTION:
    This module provides a comprehensive system for recording and storing mouse and keyboard 
    events during designated training sessions. It captures user interactions including mouse 
    movements, clicks, scrolls, and keyboard presses/releases for later analysis.

FEATURES:
    - Real-time monitoring of mouse and keyboard events
    - Session-based recording with unique session IDs
    - Hotkey-triggered start/stop control (Ctrl+Alt+T)
    - Automatic event buffering and file persistence
    - Mouse move sampling to prevent excessive data collection
    - JSON format storage for easy data analysis
    - Session metadata tracking (start time, end time, etc.)

ARCHITECTURE:
    - Event Classes: Structured representation of different event types
        * MouseMoveEvent, MouseClickEvent, MouseScrollEvent
        * KeyEvent (press/release)
    - ApplicationState: Manages global application state (training mode, stop flag)
    - EventRecorder: Core recorder class that handles event listening and storage

WORKFLOW:
    1. EventRecorder runs continuously in the background, monitoring all input events
    2. When training mode is OFF: Events are detected but not recorded
    3. When training mode is ON (Ctrl+Alt+T): Events are captured and stored in buffer
    4. Buffer automatically flushes to JSON file when reaching 1000 events
    5. On session end (Ctrl+Alt+T again): Remaining events are saved to disk

OUTPUT FILES:
    recorded_data/
    ├── session_<timestamp>.json           # Event data in JSON format
    └── session_<timestamp>_metadata.json  # Session information (start/end time)

USAGE:
    1. Run the script: python event_recorder.py
    2. Press Ctrl+Alt+T to start recording
    3. Perform actions (mouse/keyboard interactions)
    4. Press Ctrl+Alt+T to stop recording
    5. Press Ctrl+C to exit the program
    6. Find recorded data in the 'recorded_data' directory
    7. Press Ctrl+Alt+V to start/stop verbose mode (see all events as they occur)

DEPENDENCIES:
    - pynput: For capturing mouse and keyboard events
    - Standard library: time, json, os, datetime, dataclasses, enum

CONFIGURATION:
    - mouse_move_sampling_rate: 0.1 seconds (adjustable in EventRecorder.__init__)
    - buffer_size: 1000 events (auto-flush threshold)
    - data_dir: 'recorded_data' (output directory)
    - verbose: False (set to True for detailed event logging)

AUTHOR: Kevin Li
DATE: 2025-10-21
VERSION: 1.0
"""


from pynput import mouse, keyboard
import time
import json
import os
import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Union
from enum import Enum


# ======== Event Class Definitions ========

class EventType(Enum):
    """
    Define 5 event types:
    - mouse: mouse_move, mouse_click, mouse_scroll
    - keyboard: key_press, key_release
    """

    MOUSE_MOVE = "MOUSE_MOVE"
    MOUSE_CLICK = "MOUSE_CLICK"
    MOUSE_SCROLL = "MOUSE_SCROLL"
    KEY_PRESS = "KEY_PRESS"
    KEY_RELEASE = "KEY_RELEASE"


@dataclass
class BaseEvent:
    """Base event class"""
    timestamp: float
    session_id: str
    event_type: EventType
    
    def to_dict(self):
        """
        Convert to dictionary format for storage
        
        Example:
         {
            "timestamp": 1234567890.123,
            "session_id": "session_1234567890",
            "event_type": "MOUSE_CLICK",
            "x": 100,
            "y": 200,
            "button": "left",
            "action": "press"
        }
        """
        data = asdict(self)
        # change from " event_type: EventType.MOUSE_CLICK "
        # to "event_type: 'MOUSE_CLICK' " for readability
        data['event_type'] = self.event_type.value
        return data
    
    def __str__(self):
        """String representation for easy printing"""
        return f"[{self.event_type.value}] at {self.timestamp:.3f}"


@dataclass
class MouseMoveEvent(BaseEvent):
    """Mouse move event"""
    x: int
    y: int
    
    def __init__(self, x, y, session_id):
        super().__init__(
            timestamp=time.time(),
            session_id=session_id,
            event_type=EventType.MOUSE_MOVE
        )
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"[MOUSE_MOVE] ({self.x}, {self.y}) at {self.timestamp:.3f}"


@dataclass
class MouseClickEvent(BaseEvent):
    """Mouse click event"""
    x: int
    y: int
    button: str  # 'left', 'right', 'middle'
    action: str  # 'press', 'release'
    
    def __init__(self, x, y, button, pressed, session_id):
        super().__init__(
            timestamp=time.time(),
            session_id=session_id,
            event_type=EventType.MOUSE_CLICK
        )
        self.x = x
        self.y = y
        self.button = button.name
        self.action = 'press' if pressed else 'release'
    
    def __str__(self):
        return f"[MOUSE_CLICK] {self.button} {self.action} at ({self.x}, {self.y})"


@dataclass
class MouseScrollEvent(BaseEvent):
    """Mouse scroll event"""
    x: int
    y: int
    dx: int
    dy: int
    direction: str  # 'up', 'down'
    
    def __init__(self, x, y, dx, dy, session_id):
        super().__init__(
            timestamp=time.time(),
            session_id=session_id,
            event_type=EventType.MOUSE_SCROLL
        )
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.direction = 'up' if dy > 0 else 'down'
    
    def __str__(self):
        return f"[MOUSE_SCROLL] {self.direction} at ({self.x}, {self.y})"


@dataclass
class KeyEvent(BaseEvent):
    """Keyboard event (press/release)"""
    key: str
    action: str  # 'press', 'release'
    
    def __init__(self, key, action, session_id, event_type):
        super().__init__(
            timestamp=time.time(),
            session_id=session_id,
            event_type=event_type
        )
        # Handle special keys
        try:
            self.key = key.char
        except AttributeError:
            self.key = str(key).replace('Key.', '')
        self.action = action
    
    def __str__(self):
        return f"[{self.event_type.value}] {self.key}"


# ======== Application State Class ========

class ApplicationState():
    """Manages the application's global state"""
    def __init__(self):
        self._training = False
        self._stop = False
    
    @property
    def training(self):
        """Check if currently in training mode"""
        return self._training
    
    @training.setter
    def training(self, value: bool):
        """Set training mode"""
        self._training = value

    @property
    def stop(self):
        """Check if application should stop"""
        return self._stop
    
    @stop.setter
    def stop(self, value: bool):
        """Set stop flag"""
        self._stop = value

    def __str__(self):
        return 'TRAINING' if self._training else 'IDLE'


# ======== EventRecorder Class ========

class EventRecorder:
    """Records mouse and keyboard events"""
    
    def __init__(self, state: ApplicationState, verbose=False):
        self.state = state
        self.verbose = verbose
        self.events_buffer = []
        self.session_id = None
        self.session_metadata = None
        self.mouse_listener = None
        self.keyboard_listener = None
        
        # Settings
        self.last_mouse_move_time = 0
        self.mouse_move_sampling_rate = 0.1  # seconds
        self.data_dir = 'recorded_data'
    

    # ---- Mouse Event Handlers ----
    
    def on_mouse_move(self, x, y):
        """Handle mouse move event"""
        if not self.state.training:
            return
        
        # Sampling control to avoid excessive events
        current_time = time.time()
        if current_time - self.last_mouse_move_time < self.mouse_move_sampling_rate:
            return
        
        # Save this moment as the latest one that a mouse_move event occurs
        self.last_mouse_move_time = current_time
        
        # Create event object
        event = MouseMoveEvent(x, y, self.session_id)
        self.save_event(event)
    
    
    def on_mouse_click(self, x, y, button, pressed):
        """Handle mouse click event"""
        if not self.state.training:
            return
        
        event = MouseClickEvent(x, y, button, pressed, self.session_id)
        self.save_event(event)

    
    def on_mouse_scroll(self, x, y, dx, dy):
        """Handle mouse scroll event"""
        if not self.state.training:
            return
        
        event = MouseScrollEvent(x, y, dx, dy, self.session_id)
        self.save_event(event)
    

     # ---- Keyboard Event Handlers ----
    
    def on_key_press(self, key):
        """Handle key press event"""
        if not self.state.training:
            return
        
        event = KeyEvent(key, 'press', self.session_id, EventType.KEY_PRESS)
        self.save_event(event)
    
    
    def on_key_release(self, key):
        """Handle key release event"""
        if not self.state.training:
            return
        
        event = KeyEvent(key, 'release', self.session_id, EventType.KEY_RELEASE)
        self.save_event(event)
    
    
    # ---- Core Methods ----
    
    def save_event(self, event: BaseEvent):
        """Add event object to buffer"""
        if self.verbose:
            print(f"Saving: {event}")
        
        self.events_buffer.append(event)
        
        # Auto-flush when buffer is too large
        if len(self.events_buffer) >= 1000:
            self.flush_to_file()
    
    
    def flush_to_file(self):
        """Write buffered events to file"""
        if not self.events_buffer:
            return
        
        try:
            print(f"Flushing {len(self.events_buffer)} events to file...")
            
            # Convert to dictionary format
            events_data = [event.to_dict() for event in self.events_buffer]
            
            # Create data directory
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Generate filename
            filename = os.path.join(self.data_dir, f"{self.session_id}.json")
            
            # Read existing data if file exists
            existing_data = []
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            
            # Append new data
            existing_data.extend(events_data)
            
            # Write back to file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            
            print(f"Successfully saved to {filename}")
            
        except Exception as e:
            print(f"Error saving events: {e}")
        
        finally:
            # Clear buffer regardless of success/failure
            self.events_buffer = []
    
    
    def start_session(self):
        """Start a new recording session"""
        timestamp = int(time.time())
        self.session_id = f"session_{timestamp}"
        self.events_buffer = []
        
        # Create session metadata
        self.session_metadata = {
            'session_id': self.session_id,
            'start_time': timestamp,
            'start_time_readable': datetime.datetime.fromtimestamp(timestamp).isoformat(),
        }
        
        print(f"Session started: {self.session_id}")
    
    
    def end_session(self):
        """End current session"""
        if self.session_id is None:
            print("Warning: No active session to end")
            return
        
        # Flush remaining events
        self.flush_to_file()
        
        # Update metadata
        if self.session_metadata:
            self.session_metadata['end_time'] = int(time.time())
            self.session_metadata['end_time_readable'] = datetime.datetime.now().isoformat()
            
            # Save metadata
            try:
                metadata_file = os.path.join(self.data_dir, f"{self.session_id}_metadata.json")
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(self.session_metadata, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Error saving metadata: {e}")
        
        print(f"Session ended: {self.session_id}")
        self.session_id = None
        self.session_metadata = None
    
    
    def run(self):
        """Start listeners (non-blocking)"""
        self.mouse_listener = mouse.Listener(
            on_move=self.on_mouse_move,
            on_click=self.on_mouse_click,
            on_scroll=self.on_mouse_scroll
        )
        
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )
        
        self.mouse_listener.start()
        self.keyboard_listener.start()

        print("Event recorder is running")
    
    
    def stop(self):
        """Stop listeners"""
        if self.mouse_listener:
            self.mouse_listener.stop()
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        print("Event recorder stopped")


# ======== Usage Example ========

# Create application state instance
app_state = ApplicationState()

# Create recorder instance with state (set verbose=True for debugging)
recorder = EventRecorder(app_state, verbose=False)


def hotkey_train_toggle():
    """Toggle training mode on/off"""
    if not app_state.training:
        recorder.start_session()
        app_state.training = True
        print('Training started')
    else:
        recorder.end_session()
        app_state.training = False
        print('Training stopped')


def hotkey_verbose_toggle():
    """Toggle verbose mode on/off"""
    if recorder.verbose:
        print('~~~ Verbose Mode Off ~~~')
        recorder.verbose = False
    else:
        print('~~~ Verbose Mode On ~~~')
        recorder.verbose = True


# Main program
if __name__ == "__main__":
    # Setup hotkeys
    hotkeys = keyboard.GlobalHotKeys({
        '<ctrl>+<alt>+t': hotkey_train_toggle,
        '<ctrl>+<alt>+v': hotkey_verbose_toggle
    })
    hotkeys.start()
    
    # Start recorder
    recorder.run()
    
    # Keep program running
    print("=" * 50)
    print("Event Recorder Ready")
    print("=" * 50)
    print("Press Ctrl+Alt+T to start/stop recording")
    print("Press Ctrl+C to exit")
    print("=" * 50)
    
    try:
        while not app_state.stop:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        if app_state.training:
            recorder.end_session()
        recorder.stop()
        hotkeys.stop()
        print("Goodbye!")