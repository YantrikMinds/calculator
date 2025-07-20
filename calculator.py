import cv2
import mediapipe as mp
import numpy as np
import math
import time
from datetime import datetime

class VirtualTouchCalculator:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Calculator state
        self.current_number = ""
        self.previous_number = ""
        self.operation = ""
        self.display = "0"
        self.history = []
        self.just_calculated = False
        
        # Touch detection
        self.finger_tip_id = 8  # Index finger tip
        self.last_touch_time = 0
        self.touch_cooldown = 0.3  # 300ms cooldown between touches
        self.touch_threshold = 30  # pixels distance for touch detection
        self.button_pressed = None
        self.button_press_time = 0
        self.press_duration = 0.2  # Visual feedback duration
        
        # UI settings
        self.theme = "dark"
        self.show_instructions = True
        
        # Button layout - calculator style
        self.button_layout = [
            ['C', '±', '%', '÷'],
            ['7', '8', '9', '×'],
            ['4', '5', '6', '-'],
            ['1', '2', '3', '+'],
            ['0', '.', '=', 'del']
        ]
        
        # Button regions (will be calculated)
        self.buttons = {}
        
        # Color themes
        self.themes = {
            "dark": {
                "bg": (30, 30, 35),
                "panel": (50, 50, 55),
                "button": (70, 70, 75),
                "button_hover": (90, 90, 95),
                "button_pressed": (110, 110, 115),
                "number_button": (60, 60, 65),
                "operator_button": (255, 149, 0),
                "special_button": (100, 100, 100),
                "text": (255, 255, 255),
                "accent": (0, 255, 255),
                "success": (0, 255, 0),
                "error": (255, 100, 100),
                "display_bg": (20, 20, 25)
            },
            "light": {
                "bg": (240, 240, 245),
                "panel": (220, 220, 225),
                "button": (200, 200, 205),
                "button_hover": (180, 180, 185),
                "button_pressed": (160, 160, 165),
                "number_button": (210, 210, 215),
                "operator_button": (255, 149, 0),
                "special_button": (150, 150, 155),
                "text": (0, 0, 0),
                "accent": (255, 100, 0),
                "success": (0, 150, 0),
                "error": (200, 0, 0),
                "display_bg": (250, 250, 255)
            }
        }
    
    def setup_buttons(self, frame_width, frame_height):
        """Setup button positions and sizes"""
        # Calculator panel dimensions
        panel_width = 400
        panel_height = frame_height
        panel_x = frame_width - panel_width
        
        # Button dimensions
        button_width = 80
        button_height = 60
        button_margin = 8
        
        # Starting position for buttons
        start_x = panel_x + 20
        start_y = 200  # Below the display
        
        self.buttons.clear()
        
        for row_idx, row in enumerate(self.button_layout):
            for col_idx, button_text in enumerate(row):
                x = start_x + col_idx * (button_width + button_margin)
                y = start_y + row_idx * (button_height + button_margin)
                
                # Special handling for '0' button (make it wider)
                if button_text == '0':
                    width = button_width * 2 + button_margin
                else:
                    width = button_width
                
                self.buttons[button_text] = {
                    'rect': (x, y, x + width, y + button_height),
                    'center': (x + width // 2, y + button_height // 2),
                    'pressed': False,
                    'hover': False
                }
    
    def get_finger_position(self, landmarks, frame_width, frame_height):
        """Get index finger tip position"""
        if landmarks:
            finger_tip = landmarks[self.finger_tip_id]
            x = int(finger_tip.x * frame_width)
            y = int(finger_tip.y * frame_height)
            return (x, y)
        return None
    
    def is_point_in_rect(self, point, rect):
        """Check if point is inside rectangle"""
        x, y = point
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def detect_button_touch(self, finger_pos):
        """Detect which button is being touched"""
        if not finger_pos:
            # Reset all hover states
            for button in self.buttons.values():
                button['hover'] = False
            return None
        
        touched_button = None
        min_distance = float('inf')
        
        # Check each button
        for button_text, button_data in self.buttons.items():
            rect = button_data['rect']
            center = button_data['center']
            
            # Check if finger is over button
            if self.is_point_in_rect(finger_pos, rect):
                distance = self.calculate_distance(finger_pos, center)
                if distance < self.touch_threshold and distance < min_distance:
                    min_distance = distance
                    touched_button = button_text
                    button_data['hover'] = True
                else:
                    button_data['hover'] = False
            else:
                button_data['hover'] = False
        
        return touched_button
    
    def is_touching_gesture(self, landmarks):
        """Detect if user is making a touching gesture (index finger extended)"""
        if not landmarks:
            return False
        
        # Check if only index finger is extended
        finger_tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        finger_pips = [3, 6, 10, 14, 18]
        
        # Index finger should be extended
        index_extended = landmarks[finger_tips[1]].y < landmarks[finger_pips[1]].y
        
        # Other fingers should be closed (except thumb which has different logic)
        middle_closed = landmarks[finger_tips[2]].y > landmarks[finger_pips[2]].y
        ring_closed = landmarks[finger_tips[3]].y > landmarks[finger_pips[3]].y
        pinky_closed = landmarks[finger_tips[4]].y > landmarks[finger_pips[4]].y
        
        return index_extended and middle_closed and ring_closed and pinky_closed
    
    def process_button_press(self, button_text):
        """Process button press with calculator logic"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_touch_time < self.touch_cooldown:
            return
        
        print(f"Button pressed: {button_text}")
        
        # Mark button as pressed for visual feedback
        if button_text in self.buttons:
            self.buttons[button_text]['pressed'] = True
            self.button_pressed = button_text
            self.button_press_time = current_time
        
        # Process the button press
        if button_text.isdigit():
            if self.just_calculated:
                self.display = button_text
                self.current_number = button_text
                self.just_calculated = False
            else:
                if self.display == "0" or "Error" in self.display:
                    self.display = button_text
                else:
                    if len(self.display) < 12:  # Limit display length
                        self.display += button_text
                self.current_number = self.display
        
        elif button_text == '.':
            if '.' not in self.display and not self.just_calculated:
                if self.display == "0":
                    self.display = "0."
                else:
                    self.display += "."
                self.current_number = self.display
        
        elif button_text in ['+', '-', '×', '÷']:
            if self.current_number:
                if self.operation and self.previous_number:
                    # Chain operations
                    result = self.calculate(self.previous_number, self.operation, self.current_number)
                    if "Error" not in result:
                        self.previous_number = result
                        self.display = result
                    else:
                        self.display = result
                        self.last_touch_time = current_time
                        return
                else:
                    self.previous_number = self.current_number
                
                self.operation = button_text
                self.current_number = ""
                self.just_calculated = False
        
        elif button_text == '=':
            if self.current_number and self.operation and self.previous_number:
                result = self.calculate(self.previous_number, self.operation, self.current_number)
                
                # Add to history
                history_entry = f"{self.previous_number} {self.operation} {self.current_number} = {result}"
                self.history.append(history_entry)
                if len(self.history) > 10:  # Keep only last 10 calculations
                    self.history.pop(0)
                
                self.display = result
                self.current_number = result if "Error" not in result else ""
                self.previous_number = ""
                self.operation = ""
                self.just_calculated = True
        
        elif button_text == 'C':
            self.current_number = ""
            self.previous_number = ""
            self.operation = ""
            self.display = "0"
            self.just_calculated = False
        
        elif button_text == 'del':
            if not self.just_calculated and len(self.display) > 1:
                self.display = self.display[:-1]
                self.current_number = self.display
            elif len(self.display) == 1:
                self.display = "0"
                self.current_number = ""
        
        elif button_text == '±':
            if self.current_number and self.current_number != "0":
                if self.current_number.startswith('-'):
                    self.current_number = self.current_number[1:]
                else:
                    self.current_number = '-' + self.current_number
                self.display = self.current_number
        
        elif button_text == '%':
            if self.current_number:
                try:
                    result = str(float(self.current_number) / 100)
                    self.display = result
                    self.current_number = result
                except:
                    self.display = "Error"
        
        self.last_touch_time = current_time
    
    def calculate(self, num1, operation, num2):
        """Perform calculation"""
        try:
            n1, n2 = float(num1), float(num2)
            
            if operation == '+':
                result = n1 + n2
            elif operation == '-':
                result = n1 - n2
            elif operation == '×':
                result = n1 * n2
            elif operation == '÷':
                if n2 == 0:
                    return "Error"
                result = n1 / n2
            else:
                return "Error"
            
            # Format result
            if abs(result) > 999999999 or abs(result) < 0.000001 and result != 0:
                return f"{result:.2e}"
            elif result == int(result):
                return str(int(result))
            else:
                return f"{result:.8f}".rstrip('0').rstrip('.')
                
        except:
            return "Error"
    
    def draw_button(self, frame, button_text, button_data):
        """Draw individual button with proper styling"""
        colors = self.themes[self.theme]
        rect = button_data['rect']
        x1, y1, x2, y2 = rect
        
        # Determine button color based on type and state
        if button_text.isdigit() or button_text == '.':
            base_color = colors['number_button']
        elif button_text in ['+', '-', '×', '÷', '=']:
            base_color = colors['operator_button']
        else:
            base_color = colors['special_button']
        
        # Apply state-based color modifications
        if button_data['pressed'] and time.time() - self.button_press_time < self.press_duration:
            button_color = colors['button_pressed']
        elif button_data['hover']:
            button_color = colors['button_hover']
        else:
            button_color = base_color
        
        # Draw button background
        cv2.rectangle(frame, (x1, y1), (x2, y2), button_color, -1)
        
        # Draw button border
        border_color = colors['accent'] if button_data['hover'] else colors['text']
        border_thickness = 3 if button_data['hover'] else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, border_thickness)
        
        # Draw button text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2 if len(button_text) == 1 else 0.8
        text_color = colors['text'] if button_text not in ['+', '-', '×', '÷', '='] else (255, 255, 255)
        
        # Center the text
        text_size = cv2.getTextSize(button_text, font, font_scale, 2)[0]
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = y1 + (y2 - y1 + text_size[1]) // 2
        
        cv2.putText(frame, button_text, (text_x, text_y), font, font_scale, text_color, 2)
        
        # Reset pressed state after duration
        if button_data['pressed'] and time.time() - self.button_press_time > self.press_duration:
            button_data['pressed'] = False
    
    def draw_calculator_interface(self, frame):
        """Draw the complete calculator interface"""
        h, w = frame.shape[:2]
        colors = self.themes[self.theme]
        
        # Setup buttons if not already done
        if not self.buttons:
            self.setup_buttons(w, h)
        
        # Calculator panel background
        panel_width = 400
        panel_x = w - panel_width
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, 0), (w, h), colors['panel'], -1)
        cv2.addWeighted(overlay, 0.95, frame, 0.05, 0, frame)
        
        # Panel border
        cv2.rectangle(frame, (panel_x, 0), (w, h), colors['accent'], 2)
        
        # Title
        cv2.putText(frame, "VIRTUAL TOUCH CALCULATOR", (panel_x + 20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['accent'], 2)
        
        # Display screen
        display_rect = (panel_x + 20, 50, w - 20, 150)
        cv2.rectangle(frame, display_rect[:2], display_rect[2:], colors['display_bg'], -1)
        cv2.rectangle(frame, display_rect[:2], display_rect[2:], colors['accent'], 3)
        
        # Display text (right-aligned)
        display_text = self.display
        if len(display_text) > 15:
            display_text = display_text[-15:]  # Show last 15 characters
        
        text_color = colors['error'] if 'Error' in display_text else colors['text']
        font_scale = 2.0 if len(display_text) <= 8 else 1.5
        
        text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)[0]
        text_x = w - 30 - text_size[0]
        text_y = 120
        
        cv2.putText(frame, display_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 3)
        
        # Current operation indicator
        if self.operation:
            op_text = f"Operation: {self.operation}"
            cv2.putText(frame, op_text, (panel_x + 25, 175), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['accent'], 1)
        
        # Draw all buttons
        for button_text, button_data in self.buttons.items():
            self.draw_button(frame, button_text, button_data)
        
        # History section
        history_y = 550
        cv2.putText(frame, "RECENT CALCULATIONS:", (panel_x + 20, history_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['accent'], 2)
        
        # Show recent history
        for i, calc in enumerate(self.history[-4:]):
            calc_text = calc[:35] + "..." if len(calc) > 35 else calc
            y_pos = history_y + 30 + i * 25
            cv2.putText(frame, calc_text, (panel_x + 25, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors['text'], 1)
    
    def draw_instructions(self, frame):
        """Draw usage instructions"""
        if not self.show_instructions:
            return
            
        colors = self.themes[self.theme]
        instructions = [
            "HOW TO USE:",
            "1. Point with INDEX finger",
            "2. Touch buttons to press them",
            "3. Keep other fingers closed",
            "",
            "KEYBOARD SHORTCUTS:",
            "Q: Quit  T: Theme  I: Instructions",
            "R: Reset History  C: Clear"
        ]
        
        # Instructions background
        inst_height = len(instructions) * 30 + 40
        cv2.rectangle(frame, (10, 10), (350, 10 + inst_height), colors['panel'], -1)
        cv2.rectangle(frame, (10, 10), (350, 10 + inst_height), colors['accent'], 2)
        
        for i, instruction in enumerate(instructions):
            color = colors['accent'] if instruction.endswith(':') else colors['text']
            font_scale = 0.6 if instruction.endswith(':') else 0.5
            thickness = 2 if instruction.endswith(':') else 1
            
            cv2.putText(frame, instruction, (20, 40 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    def draw_finger_tracking(self, frame, landmarks, finger_pos):
        """Draw finger tracking visualization"""
        if not landmarks:
            return
            
        colors = self.themes[self.theme]
        
        # Draw hand skeleton
        connections = self.mp_hands.HAND_CONNECTIONS
        for connection in connections:
            start_idx, end_idx = connection
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            start_pos = (int(start.x * frame.shape[1]), int(start.y * frame.shape[0]))
            end_pos = (int(end.x * frame.shape[1]), int(end.y * frame.shape[0]))
            
            cv2.line(frame, start_pos, end_pos, colors['accent'], 2)
        
        # Highlight finger joints
        for i, landmark in enumerate(landmarks):
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            
            if i == self.finger_tip_id:  # Index finger tip
                cv2.circle(frame, (x, y), 12, colors['success'], -1)
                cv2.circle(frame, (x, y), 12, (255, 255, 255), 3)
            else:
                cv2.circle(frame, (x, y), 6, colors['accent'], -1)
        
        # Draw touch indicator
        if finger_pos:
            cv2.circle(frame, finger_pos, 25, colors['success'], 2)
    
    def run(self):
        """Main application loop"""
        print("Starting Virtual Touch Calculator...")
        print("Point with your INDEX finger and touch the virtual buttons!")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Camera ready! Keep other fingers closed, point with index finger.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            finger_pos = None
            touching = False
            
            # Process hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = hand_landmarks.landmark
                    
                    # Get finger position
                    finger_pos = self.get_finger_position(landmarks, w, h)
                    
                    # Check if making touching gesture
                    touching = self.is_touching_gesture(landmarks)
                    
                    # Draw hand tracking
                    self.draw_finger_tracking(frame, landmarks, finger_pos)
            
            # Detect button interaction
            touched_button = self.detect_button_touch(finger_pos)
            
            # Process touch if touching gesture is detected
            if touching and touched_button:
                self.process_button_press(touched_button)
            
            # Draw interface
            self.draw_calculator_interface(frame)
            self.draw_instructions(frame)
            
            # Status display
            colors = self.themes[self.theme]
            status = "TOUCHING" if touching and touched_button else "POINTING" if finger_pos else "NO HAND"
            status_color = colors['success'] if touching else colors['accent'] if finger_pos else colors['error']
            
            cv2.putText(frame, f"STATUS: {status}", (10, h - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            if touched_button:
                cv2.putText(frame, f"HOVERING: {touched_button}", (10, h - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['accent'], 2)
            
            # Show frame
            cv2.imshow('Virtual Touch Calculator - Touch buttons with your finger!', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('t') or key == ord('T'):
                self.theme = "light" if self.theme == "dark" else "dark"
            elif key == ord('i') or key == ord('I'):
                self.show_instructions = not self.show_instructions
            elif key == ord('r') or key == ord('R'):
                self.history.clear()
            elif key == ord('c') or key == ord('C'):
                self.process_button_press('C')
        
        cap.release()
        cv2.destroyAllWindows()
        print("Calculator closed!")

def main():
    try:
        calculator = VirtualTouchCalculator()
        calculator.run()
    except KeyboardInterrupt:
        print("\nCalculator stopped by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()