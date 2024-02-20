from tracker.eye_tracker import EyeTrackerFunction
from tracker.hand_tracker import HandTrackerFunction
from listener.listener import TrackerListener
from controller.controll_event import handle_events, handle_events_virtual_mouse
if __name__ == '__main__':
    eye_tracker_function = EyeTrackerFunction.get_tracker_function(EyeTrackerFunction.DL_EYE_TRACKING)
    hand_tracker_function = HandTrackerFunction.get_tracker_function(HandTrackerFunction.BASE_METHOD)
    tracker_listener = TrackerListener(tracker_functions=[eye_tracker_function, hand_tracker_function],
                                        control_function=handle_events_virtual_mouse)
    
    tracker_listener.start()
