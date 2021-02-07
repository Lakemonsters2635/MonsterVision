import cv2
import depthai
import sys
import time
from networktables import NetworkTables
from cscore import CameraServer
import json
import socket

configFile = "/boot/frc.json"


# To see messages from networktables, you must setup logging
import logging

logging.basicConfig(level=logging.DEBUG)

team = None
server = False

def getMyIP():
    return socket.gethostbyname(socket.gethostname())

def parseError(str):
    """Report parse error."""
    print("config error in '" + configFile + "': " + str, file=sys.stderr)


def readConfig():
    global team
    global server

    try:
        with open(configFile, "rt", encoding="utf-8") as f:
            j = json.load(f)
    except OSError as err:
        print("could not open '{}': {}".format(configFile, err), file=sys.stderr)
        return False
        
    # top level must be an object
    if not isinstance(j, dict):
        parseError("must be JSON object")
        return False

    # team number
    try:
        team = j["team"]
    except KeyError:
        parseError("could not read team number")
        return False

    # ntmode (optional)
    if "ntmode" in j:
        str = j["ntmode"]
        if str.lower() == "client":
            server = False
        elif str.lower() == "server":
            server = True
        else:
            parseError("could not understand ntmode value '{}'".format(str))
    
    return True

if not readConfig():
    sys.exit(1)

if server:
    NT_SERVER_ADDRESS = getMyIP()
else:
    NT_SERVER_ADDRESS = "10." + str(int(team/100)) + "." + str(int(team%100)) + ".2"

ip=NT_SERVER_ADDRESS
NetworkTables.initialize(server=ip)

sd = NetworkTables.getTable("SmartDashboard")

# cv2.namedWindow('MonsterVision', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('MonsterVision', (600, 600))

# Initialize the OAK Camera.  This is boilerplate.
cameraFPS = 30
desiredFPS = 5
width = 200
height = 200
cs = CameraServer.getInstance()
cs.enableLogging()
output = cs.putVideo("MonsterVision", width, height)

device = depthai.Device('', False)

p = device.create_pipeline(config={
    "streams": ["metaout", "previewout"
            # , "disparity_color"                 ## Enable this to see false color depth map display
        ],
    "ai": {
# The next five lines determine which model to use.
        "blob_file": "./resources/nn/cells-and-cones/cells-and-cones.blob.sh14cmx14NCE1",
        "blob_file_config": "./resources/nn/cells-and-cones/cells-and-cones.json",
        # "blob_file": "./resources/nn/trafficcones/trafficcones.blob.sh14cmx14NCE1",
        # "blob_file_config": "./resources/nn/trafficcones/trafficcones.json",
        'shaves' : 14,
        'cmx_slices' : 14,
        'NN_engines' : 1,
# This line is needed enable the depth feature
        'calc_dist_to_bb' : True,
    },
    "depth": {
        'padding_factor': 0.3
    }
})

if p is None:
    raise RuntimeError("Error initializing pipelne")

# Start with an empty list of detections.  This is to avoid an undefined variable while waiting for
# the first nnet packets to arrive.

detections = []
frame_counter = 0
while True:
# The pipeline returns nnet packets and data packets.  Nnet packets contain the output(s) of the
# neural network (e.g. bounding box, confidence, label for object detection).  The data packets
# contain image data

    nnet_packets, data_packets = p.get_available_nnet_and_data_packets()

# Extract the nnet packets into a list.  For object detection, there is one nn packet for each instance
# on an object found.  Note that the FPS of the neural network engine may not be the same as that of
# the camera (slower).  If data packets arrive without any nnet packets, the detections = ...
# assignment statement is not executed.  That's why detections was initialized to empty above.
# The result is that if nnet_packets contains NO packets, we reuse the previous one when drawing on
# the image.  If it contains multiple, we use only the last one.

    for nnet_packet in nnet_packets:
        detections = list(nnet_packet.getDetectedObjects())

# Loop over the data packets.
    for packet in data_packets:

        if packet.stream_name == "disparity_color":
            data = packet.getData()
            # data0 = data[0, :, :]
            # data1 = data[1, :, :]
            # data2 = data[2, :, :]
            # frame = cv2.merge([data0, data1, data2])
            cv2.imshow('Depth', data)
           
            
# The previewout stream is our conventional video stream
# NOTE: we can only get video data from the camera via the DepthAI API.  This means we cannot use any
# # standard Linux camera s/w to view the OAK video.
#         
        if packet.stream_name == 'previewout':
    
# The camera returns data in RGB format.  OpenCV use BGR.  These lines create an OpenCV-compatible
# image.  Unless we want to preview the RGB stream, this step would not be needed on the robot.

            data = packet.getData()
            data0 = data[0, :, :]
            data1 = data[1, :, :]
            data2 = data[2, :, :]
            frame = cv2.merge([data0, data1, data2])

# Get the height and width of the video frame.  We need this because the bounding box is returned as
# a fraction of the overall image size.  So we use these value to convert to pixels.

            img_h = frame.shape[0]
            img_w = frame.shape[1]

            sd.putNumber("MonsterVision/Numberofobjects",len(detections))

            i = 0
            for detection in sorted(detections, key=lambda det: det.label*100 + det.depth_z):

# Pt2 and pt2 define the bounding box.  Create them from (x_min, x_min) and (x_max, y_max).  Note how
# we scale them to pixels as mentioned above.

                pt1 = int(detection.x_min * img_w), int(detection.y_min * img_h)
                pt2 = int(detection.x_max * img_w), int(detection.y_max * img_h)

# Choose the color based on the label

                if detection.label == 0:
                    color = (0, 255, 255)
                else :
                    color = (20, 50, 239)

# Draw the bounding box on the image.

                cv2.rectangle(frame, pt1, pt2, color, 1)

# Ptx, pty and ptz are simply where we draw the x, y and z positions underneath the bounding box.

# If labelling would be off the bottom of the image, move to above the bounding box

                if (int(detection.y_max * img_h+50)) > img_h:
                    ptx = int(detection.x_min * img_w), int(detection.y_min * img_h-50)
                    pty = int(detection.x_min * img_w), int(detection.y_min * img_h-40)
                    ptz = int(detection.x_min * img_w), int(detection.y_min * img_h-30)
                    ptc = int(detection.x_min * img_w), int(detection.y_min * img_h-20)
                else:
                    ptx = int(detection.x_min * img_w), int(detection.y_max * img_h+10)
                    pty = int(detection.x_min * img_w), int(detection.y_max * img_h+20)
                    ptz = int(detection.x_min * img_w), int(detection.y_max * img_h+30)
                    ptc = int(detection.x_min * img_w), int(detection.y_max * img_h+40)

# Scribble the results onto the image

                cv2.putText(frame, str(i), pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))

                cv2.putText(frame, "x: " + '{:.2f}'.format(detection.depth_x*39.3071), ptx, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color)
                cv2.putText(frame, "y: " + '{:.2f}'.format(detection.depth_y*39.3071), pty, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color)
                cv2.putText(frame, "z: " + '{:.2f}'.format(detection.depth_z*39.3071), ptz, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color)
                cv2.putText(frame, '{:.2f}'.format(100*detection.confidence)+'%', ptc, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color)
                
                sd.putNumber("MonsterVision/"+str(i)+"/Label",detection.label)
                sd.putNumber("MonsterVision/"+str(i)+"/x",detection.depth_x)
                sd.putNumber("MonsterVision/"+str(i)+"/y",detection.depth_y)
                sd.putNumber("MonsterVision/"+str(i)+"/z",detection.depth_z)
                i = i+1


# Display the Frame

            # cv2.imshow('MonsterVision', frame)
            if frame_counter % (cameraFPS/desiredFPS) == 0:
                output.putFrame(frame)
            frame_counter += 1

# When user typwa 'q', we're all done!

    # if cv2.waitKey(1) == ord('q'):
    #     break

# All done.  Clean up after yourself, like your mama taught you.

del p
del device