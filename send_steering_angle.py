import serial
import time
from collections import deque

# COM port where Arduino is connected
arduino = serial.Serial('COM8', 9600, timeout=1)
time.sleep(2)  # wait for Arduino to reboot

# Angle filtering
angle_window = deque(maxlen=5)
last_sent_angle = 0
ANGLE_THRESHOLD = 3  # Ignore changes smaller than this
MAX_ANGLE_JUMP = 80  # Ignore sudden large spikes


def send_angle(new_angle):
    global last_sent_angle

    # Clamp angle to safe range
    new_angle = max(-360, min(360, new_angle))

    # Add to rolling window
    angle_window.append(new_angle)

    # Smooth angle
    avg_angle = sum(angle_window) / len(angle_window)

    # Ignore jitter (small change) - BUT always send first angle or when crossing zero
    diff = abs(avg_angle - last_sent_angle)
    crossing_zero = (last_sent_angle * avg_angle < 0) or (last_sent_angle == 0 and avg_angle != 0)
    
    if diff < ANGLE_THRESHOLD and not crossing_zero:
        return

    # Ignore huge spikes (likely noise)
    if diff > MAX_ANGLE_JUMP and not crossing_zero:
        print(f" Ignored large spike: {avg_angle:.2f}")
        return

    try:
        angle_str = f"{avg_angle:.2f}\n"
        arduino.write(angle_str.encode('utf-8'))
        last_sent_angle = avg_angle
        print(f" Sent angle: {avg_angle:.2f}")
    except:
        print("Could not send angle")
