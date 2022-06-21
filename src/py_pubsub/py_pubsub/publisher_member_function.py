# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
from .submodule.darknet import darknet
import time
import requests
from ctypes import *
import math
import random
import os
win_title = 'YOLOv4 CUSTOM DETECTOR'
cfg_file = 'cfg/yolov4-tiny.cfg'
data_file = 'cfg/coco.data'
weight_file = 'yolov4-tiny.weights'
thre = 0.25
show_coordinates = True
def print_detections(detections, coordinates=False):
    print("\nObjects:")
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        if coordinates:
            print("{}: {}%    (left_x: {:.0f}   top_y:  {:.0f}   width:   {:.0f}   height:  {:.0f})".format(label, confidence, x, y, w, h))
        else:
            print("{}: {}%".format(label, confidence))

# Load Network
network, class_names, class_colors = darknet.load_network(
        cfg_file,
        data_file,
        weight_file,
        batch_size=1
    )

# Get Nets Input dimentions

width = darknet.network_width(network)
height = darknet.network_height(network)
# Video Stream
cap = cv2.VideoCapture("/dev/video0")

#ros node
class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
    def detections(self):
        for label, confidence, bbox in detections:
            bbox.data = "bbox:" + bbox
            self.publisher_.publish(bbox)
            self.get_logger().info('Publishing: "%s"' % bbox.data)
    """def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1"""


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    # minimal_publisher.destroy_node()
    # rclpy.shutdown()


if __name__ == '__main__':
	while cap.isOpened():

		# Get current frame, quit if no frame 

		    ret, frame = cap.read()

		    if not ret: break

		    t_prev = time.time()
		# Fix image format

		    frame_rgb = cv2.cvtColor( frame, cv2.COLOR_BGR2RGB)
		    frame_resized = cv2.resize( frame_rgb, (width, height))
		# convert to darknet format, save to " darknet_image "

		    darknet_image = darknet.make_image(width, height, 3)
		    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes()) 
		# inference

		    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thre)
		    darknet.print_detections(detections, show_coordinates)
		    darknet.free_image(darknet_image)
		# draw bounding box

		    image = darknet.draw_boxes(detections, frame_resized, class_colors)
		    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		#print detections
		    darknet.print_detections(detections, '--ext_output')
		# Show Image and FPS

		    fps = int(1/(time.time()-t_prev))
		    cv2.rectangle(image, (5, 5), (75, 25), (0,0,0), -1)
		    cv2.putText(image, f'FPS {fps}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		    cv2.imshow(win_title, image)
		    if cv2.waitKey(1) == ord('q'):
		       break
		    cv2.destroyAllWindows()
		    cap.release()
		    main()
