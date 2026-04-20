# Description: Example of how to use the Tobii 4C or 5 using the GazeTracker class.
from GazeTracker import GazeTracker
import time

# Start the eye tracker
tracker = GazeTracker()

for i in range(100):
    # Read data every second, result is a list of tuples (x, y)
    points = tracker.get_movement()
    print(points)
    with open("data.txt", "a") as f:
        for x, y in points:
            f.write(f"{x},{y}\n")


    time.sleep(1)



# deprecated example for win32


# import TobiiEyeTracker

# # Start the eye tracker
# try:
#     TobiiEyeTracker.init()
# except:
#     pass

# # Read data every second
# import time
# for i in range(100):
#     print(TobiiEyeTracker.getBuffer())
#     time.sleep(1)
    