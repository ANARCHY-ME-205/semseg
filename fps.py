#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import time

class ImageCapturePublish:
    def __init__(self):
        rospy.init_node('image_capture_publish_node', anonymous=True)
        
        self.source_topic = '/zed2i/zed_node/rgb/image_rect_color'  # Change this to your source topic
        self.target_topic = '/lowfps'  # Change this to your target topic
        
        self.image_sub = rospy.Subscriber(self.source_topic, Image, self.image_callback)
        self.image_pub = rospy.Publisher(self.target_topic, Image, queue_size=10)
        
        self.interval = 0.15  # 2 seconds interval
        self.last_capture_time = rospy.get_time()
        
    def image_callback(self, msg):
        current_time = rospy.get_time()
        if current_time - self.last_capture_time >= self.interval:
            self.last_capture_time = current_time
            self.publish_image(msg)
            
    def publish_image(self, img_msg):
        self.image_pub.publish(img_msg)
        rospy.loginfo("Published an image")

if __name__ == '__main__':
    try:
        image_capture_publish = ImageCapturePublish()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
