import eventlet
import socketio
from flask import Flask
import signal
import sys

# Create SocketIO server
sio = socketio.Server()
app = Flask(__name__)

@sio.on('connect')
def connect(sid, environ):
    print("✅ Connected to simulator:", sid)
    # Send initial throttle as soon as connected
    send_control(0.0, 0.3, 0.0)

@sio.on('telemetry')
def telemetry(sid, data):
    # Always accelerate straight with throttle=0.3
    send_control(0.0, 0.3, 0.0)

def send_control(steering, throttle, brake):
    sio.emit("steer", data={
        'steering_angle': str(steering),
        'throttle': str(throttle),
        'brake': str(brake)
    })

# Handle CTRL+C exit
def signal_handler(sig, frame):
    print("\n🛑 Exiting safely...")
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    print("🚗 Waiting for simulator... (Press CTRL+C to stop)")
    app = socketio.WSGIApp(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
