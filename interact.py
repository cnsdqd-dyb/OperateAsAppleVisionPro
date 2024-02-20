from tracker.eye_tracker import EyeTrackerFunction
from tracker.hand_tracker import HandTrackerFunction
from listener.listener import TrackerListener
from controller.controll_event import handle_events, handle_events_virtual_mouse
import json, os

class MoveAsAppleVisionPro:
    def __init__(self):
        if not os.path.exists('config.json'):
            with open('config.json', 'w') as f:
                user_name = input("Input Your Name:")
                json.dump({'name': user_name}, f)
        self.user_config = json.load(open('config.json', 'r'))

        # Set the eye tracker function
        input_selection = input("Select the eye tracker function: 1. Deep Learning Eye Tracking, 2. Random Forest Eye Tracking, 3. Adjust Eye Tracking Method")
        if input_selection == '1':
            self.eye_tracker_function = EyeTrackerFunction.get_tracker_function(EyeTrackerFunction.DL_EYE_TRACKING)
        elif input_selection == '2':
            self.eye_tracker_function = EyeTrackerFunction.get_tracker_function(EyeTrackerFunction.RF_EYE_TRACKING)
        elif input_selection == '3':
            self.eye_tracker_function = EyeTrackerFunction.get_tracker_function(EyeTrackerFunction.ADJUST_EYE_TRACKING)
        else:
            raise ValueError("Invalid input")
        
        # query if the user wants to use the hand tracker
        input_selection = input("Do you want to use the hand tracker? (y/n)")
        if input_selection == 'y':
            self.hand_tracker_function = HandTrackerFunction.get_tracker_function(HandTrackerFunction.BASE_METHOD)
        else:
            self.hand_tracker_function = None
        
        # select the control function
        input_selection = input("Select the control function: 1. Virtual Mouse, 2. Real Mouse")
        if input_selection == '1':
            self.control_function = handle_events_virtual_mouse
        elif input_selection == '2':
            self.control_function = handle_events
        else:
            raise ValueError("Invalid input")
        
        tracker_functions = []
        if self.eye_tracker_function:
            tracker_functions.append(self.eye_tracker_function)
        if self.hand_tracker_function:
            tracker_functions.append(self.hand_tracker_function)
        self.tracker_listener = TrackerListener(tracker_functions=tracker_functions,
                                                control_function=self.control_function)
    
    def start(self):
        self.tracker_listener.start()

        # ESC to stop the program
        
if __name__ == '__main__':
    move_as_apple_vision_pro = MoveAsAppleVisionPro()
    move_as_apple_vision_pro.start()