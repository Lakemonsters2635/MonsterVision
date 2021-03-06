import cv2
import depthai
#import cv_videosource as cvv
import sys
import numpy as np
import sys
import time
from networktables import NetworkTables

# To see messages from networktables, you must setup logging
import logging

CAMERA_SERVER = False
LIVE_VIDEO = True

logging.basicConfig(level=logging.DEBUG)

ip="192.168.1.53"
NetworkTables.initialize(server=ip)

sd = NetworkTables.getTable("MonsterVision")

# Given an x/y coordinate in NN (which are always in the range of 0..1), return its
# pixel coordinates in the depth (disparity) frame.  nn2depth is a cached copy of the
# result of device.get_nn_to_depth_bbox_mapping

def nn_to_depth_coord(x, y, nn2depth):
    x_depth = int(nn2depth['off_x'] + x * nn2depth['max_w'])
    y_depth = int(nn2depth['off_y'] + y * nn2depth['max_h'])
    return x_depth, y_depth


# Given an object's bounding box (pt1, pt2) and the padding_factor, return the bounding
# box over which the NN has computed depth

def average_depth_coord(pt1, pt2, padding_factor):
    factor = 1 - padding_factor
    x_shift = int((pt2[0] - pt1[0]) * factor / 2)
    y_shift = int((pt2[1] - pt1[1]) * factor / 2)
    avg_pt1 = (pt1[0] + x_shift), (pt1[1] + y_shift)
    avg_pt2 = (pt2[0] - x_shift), (pt2[1] - y_shift)
    return avg_pt1, avg_pt2


if LIVE_VIDEO:    
    cv2.namedWindow('MonsterVision', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('MonsterVision', (600, 600))

if CAMERA_SERVER:
    vidsrc = cvv.CVVideoSource(fps=5)#cam_size=(640, 480))#image="fish.jpg")
    vidsrc.go()

config = {
    "streams": ["metaout", "previewout"
            , "disparity_color"                 ## Enable this to see false color depth map display
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
# Enable the following to use full RGB FOV, not keeping aspect ratio
        # 'keep_aspect_ratio' : False,
    },
    "depth": {
        'padding_factor': 0.3
    }
}

# Initialize the OAK Camera.  This is boilerplate.

device = depthai.Device('', False)

p = device.create_pipeline(config=config)

if p is None:
    raise RuntimeError("Error initializing pipelne")

# Start with an empty list of detections.  This is to avoid an undefined variable while waiting for
# the first nnet packets to arrive.

detections = []

# Since the RGB camera has a 4K resolution and the neural networks accept only images with specific
# resolution (like 300x300), the original image is cropped to meet the neural network requirements.
# On the other side, the disparity frames returned by the neural network are in full resolution 
# available on the mono cameras.
#
# nn2depth will contain:
#   max_h       height of nn previewout frame
#   max_w       width of nn previewout frame
#   offset_x    X-offset of previewout frame within disparity frame
#   offset_y    Y-offset of previewout frame within disparity frame

nn2depth = device.get_nn_to_depth_bbox_mapping()

frameNumber = 0

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
            dsp_h = data.shape[0]
            dsp_w = data.shape[1]

            for detection in detections:
    
# Pt2 and pt2 define the bounding box.  Create them from (x_min, x_min) and (x_max, y_max).  
# The call to nn_to_depth_coord converts the coordinate system

                pt1 = nn_to_depth_coord(detection.x_min, detection.y_min, nn2depth)
                pt2 = nn_to_depth_coord(detection.x_max, detection.y_max, nn2depth)

                cv2.rectangle(data, pt1, pt2, (255, 255, 255), 2)

# Avg_pt1 and avg_pts are the corners of the bounding box that the NN used to calculate the
# distance to the detected object.                

                avg_pt1, avg_pt2 = average_depth_coord(pt1, pt2, config['depth']['padding_factor'])
                cv2.rectangle(data, avg_pt1, avg_pt2, (255, 0, 0), 1)

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

            for detection in detections:

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

# Avg_pt1 and avg_pts are the corners of the bounding box that the NN used to calculate the
# distance to the detected object.                

                avg_pt1, avg_pt2 = average_depth_coord(pt1, pt2, config['depth']['padding_factor'])
                cv2.rectangle(frame, avg_pt1, avg_pt2, (255, 0, 0), 1)

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

            cv2.putText(frame, str(frameNumber), (int(img_w/2), int(img_h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0))
            frameNumber += 1

            if CAMERA_SERVER:
# I have NO idea why it is necessary to roll the frame to make it look right in the camera stream...
                vidsrc.new_frame(frame)
                # vidsrc.new_frame(np.roll(frame, (-255, 32), axis=(1, 0)))

            if LIVE_VIDEO:
                cv2.imshow('MonsterVision', frame)

# When user types 'q', we're all done!

    if LIVE_VIDEO:
        if cv2.waitKey(1) == ord('q'):
            break

# All done.  Clean up after yourself, like your mama taught you.

if CAMERA_SERVER:
    vidsrc.stop()
    del vidsrc
del p
del device
sys.exit()