import asyncio
import websockets
import json
async def connect_to_gaze_tracker():
    uri = "ws://localhost:8765"  # Match your server port
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to Gaze Tracker Server")
            while True:
                # Receive gaze data
                data = await websocket.recv()
                gaze_data = json.loads(data)
                # Process the data
                print(f"Gaze Position: X={gaze_data['x']:.2f}, Y={gaze_data['y']:.2f}")
                print(f"Timestamp: {gaze_data['timestamp']}")
    except websockets.exceptions.ConnectionClosed:
        print("Connection to server closed")
    except Exception as e:
        print(f"Error: {e}")
# Run the client
asyncio.get_event_loop().run_until_complete(connect_to_gaze_tracker())







