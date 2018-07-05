from threading import Thread, Event
import os
import stat
import json
import numpy as np
import awscam
import cv2
import mo
import greengrasssdk
import datetime
from random import randrange, uniform
import glob
import csv

class LocalDisplay(Thread):
    def __init__(self, resolution):
        """ resolution - Desired resolution of the project stream """
        # Initialize the base class, so that the object can run on its own
        # thread.
        super(LocalDisplay, self).__init__()
        # List of valid resolutions
        RESOLUTION = {'1080p' : (1920, 1080), '720p' : (1280, 720), '480p' : (858, 480)}
        if resolution not in RESOLUTION:
            raise Exception("Invalid resolution")
        self.resolution = RESOLUTION[resolution]
        # Initialize the default image to be a white canvas. Clients
        # will update the image when ready.
        self.frame = cv2.imencode('.jpg', 255*np.ones([640, 480, 3]))[1]
        self.stop_request = Event()

    def run(self):
        """ Overridden method that continually dumps images to the desired
            FIFO file.
        """
        # Path to the FIFO file. The lambda only has permissions to the tmp
        # directory. Pointing to a FIFO file in another directory
        # will cause the lambda to crash.
        result_path = '/tmp/results.mjpeg'
        # Create the FIFO file if it doesn't exist.
        if not os.path.exists(result_path):
            os.mkfifo(result_path)
        # This call will block until a consumer is available
        with open(result_path, 'w') as fifo_file:
            while not self.stop_request.isSet():
                try:
                    # Write the data to the FIFO file. This call will block
                    # meaning the code will come to a halt here until a consumer
                    # is available.
                    fifo_file.write(self.frame.tobytes())
                except IOError:
                    continue

    def set_frame_data(self, frame):
        """ Method updates the image data. This currently encodes the
            numpy array to jpg but can be modified to support other encodings.
            frame - Numpy array containing the image data of the next frame
                    in the project stream.
        """
        ret, jpeg = cv2.imencode('.jpg', cv2.resize(frame, self.resolution))
        if not ret:
            raise Exception('Failed to set frame data')
        self.frame = jpeg

    def join(self):
        self.stop_request.set()

def catcritter_infinite_infer_run():
    """ Entry point of the lambda function"""
    try:
        # captured ssd image info for training
        ssd_image_list = [];

        class_labels = {0: 'buddy', 1: 'jade', 2: 'buddypluscritter', 3: 'nothing'}

        # This object detection model is implemented as single shot detector (ssd), since
        # the number of labels is small we create a dictionary that will help us convert
        # the machine labels to human readable labels.
        output_map = {1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus',
                      7 : 'car', 8 : 'cat', 9 : 'chair', 10 : 'cow', 11 : 'dinning table',
                      12 : 'dog', 13 : 'horse', 14 : 'motorbike', 15 : 'person',
                      16 : 'pottedplant', 17 : 'sheep', 18 : 'sofa', 19 : 'train',
                      20 : 'tvmonitor'}
        # Create an IoT client for sending to messages to the cloud.
        client = greengrasssdk.client('iot-data')
        iot_topic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
        # Create a local display instance that will dump the image bytes to a FIFO
        # file that the image can be rendered locally.
        local_display = LocalDisplay('480p')
        local_display.start()

        # The height and width of the training set images
        class_input_height = 100
        class_input_width = 100
        ssd_input_height = 300
        ssd_input_width = 300

        ssd_model_type = "ssd"
        class_model_type = "classification"
        class_model_name = "image-classification"

        client.publish(topic=iot_topic, payload='optimizing model')
        error, class_model_path = mo.optimize(class_model_name,class_input_width,class_input_height, aux_inputs={'--epoch': 100})

        # The aux_inputs is equal to the number of epochs and in this case, it is 100
        # Load model to GPU (use {"GPU": 0} for CPU)
        mcfg = {"GPU": 1}

        # The sample projects come with optimized artifacts, hence only the artifact
        # path is required.
        ssd_model_path = '/opt/awscam/artifacts/mxnet_deploy_ssd_resnet50_300_FP16_FUSED.xml'
        # Load the model onto the GPU
        client.publish(topic=iot_topic, payload='Loading object detection model')
        ssd_model = awscam.Model(ssd_model_path, mcfg)
        class_model = awscam.Model(class_model_path, mcfg)
        client.publish(topic=iot_topic, payload='Object detection model loaded')
        
        # Set the threshold for detection
        detection_threshold = 0.25

        counter = 1;
        ssd_counter = 1;
        irand = randrange(0, 1000)
        num_classes = 4

        # prepare training csv
        if not os.path.isdir("/tmp/cats"):
            os.mkdir("/tmp/cats")
            os.mkdir("/tmp/cats/train")
            os.chmod("/tmp/cats", stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            os.chmod("/tmp/cats/train", stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

        if not os.path.isfile("/tmp/cats/train/train.csv"):
            with open('/tmp/cats/train/train.csv', 'a') as outcsv:
                writer = csv.writer(outcsv)
                writer.writerow(['frame','xmin','xmax','ymin','ymax','class_id'])
                outcsv.close()

        today = datetime.datetime.now().strftime("%Y%m%d")

        # Do inference until the lambda is killed.
        while True:
            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()
            if not ret:
                raise Exception('Failed to get frame from the stream')
            # Resize frame to the same size as the training set.
            frame_resize = cv2.resize(frame, (ssd_input_height, ssd_input_width))
            # Run the images through the inference engine and parse the results using
            # the parser API, note it is possible to get the output of doInference
            # and do the parsing manually, but since it is a ssd model,
            # a simple API is provided.
            parsed_inference_results = ssd_model.parseResult(ssd_model_type,ssd_model.doInference(frame_resize))
            #client.publish(topic=iot_topic, payload='ssd infer complete')

            # Compute the scale in order to draw bounding boxes on the full resolution
            # image.
            yscale = float(frame.shape[0]/ssd_input_height)
            xscale = float(frame.shape[1]/ssd_input_width)
            # Dictionary to be filled with labels and probabilities for MQTT
            cloud_output = {}

            image_saved = False;

    # Get the detected objects and probabilities
            for obj in parsed_inference_results[ssd_model_type]:
                if obj['prob'] > detection_threshold:

                    # Add bounding boxes to full resolution frame
                    xmin = int(xscale * obj['xmin']) \
                           + int((obj['xmin'] - ssd_input_width/2) + ssd_input_width/2)
                    ymin = int(yscale * obj['ymin'])
                    xmax = int(xscale * obj['xmax']) \
                           + int((obj['xmax'] - ssd_input_width/2) + ssd_input_width/2)
                    ymax = int(yscale * obj['ymax'])


                    # if we found a cat, then save the image to a file and publish to IOT
                    if obj['label'] == 8:
                        crop = frame[ymin:ymax,xmin:xmax].copy()
                        crop_resize = cv2.resize(crop, (class_input_height, class_input_width))

                        # Run model inference on the cropped frame
                        inferOutput = class_model.doInference(crop_resize)
                        #client.publish(topic=iot_topic, payload='classification infer complete')

                        class_inference_results = class_model.parseResult(class_model_type,inferOutput)
                        top_k = class_inference_results[class_model_type][0:num_classes]
                        first = top_k[0]
                        if first['prob'] > detection_threshold:
                            if first['label'] < 4:
                                #client.publish(topic=iot_topic, payload='found {}, saving file'.format(labels[first['label']]))
                                path = "/tmp/cats/{}_{:03d}_{}_{:03d}.jpg".format(today, counter, class_labels[first['label']], irand)
                                cv2.imwrite(path, crop)
                                os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
                                counter += 1
            
                                msg = '{'
                                prob_num = 0
                                for kitty in top_k:
                                    if prob_num == num_classes-1:
                                        msg += '"{}": {:.2f}'.format(class_labels[kitty["label"]], kitty["prob"]*100)
                                    else:
                                        msg += '"{}": {:.2f},'.format(class_labels[kitty["label"]], kitty["prob"]*100)
            
                                prob_num += 1
                                msg += "}"


                                # Send results to the cloud
                                client.publish(topic=iot_topic, payload=json.dumps(msg))

                                if not image_saved:
                                    frame_filename = "{}_{:03d}_{}_{:03d}.jpg".format(today, ssd_counter, 'cats', irand)
                                    frame_path = "/tmp/cats/train/{}".format(frame_filename)
                                    cv2.imwrite(frame_path, frame_resize)
                                    ssd_counter += 1
                                    image_saved = True;

                                # create ssd entry
                                ssd_image_desc = [frame_filename, int(round(obj['xmin'])), int(round(obj['xmax'])), int(round(obj['ymin'])), int(round(obj['ymax'])), first['label']+1]
                                ssd_image_list.append(ssd_image_desc)

                    elif obj['label'] == 15:
                        if not image_saved:
                            frame_filename = "{}_{:03d}_{}_{:03d}.jpg".format(today, ssd_counter, 'person', irand)
                            frame_path = "/tmp/cats/train/{}".format(frame_filename)
                            cv2.imwrite(frame_path, frame_resize)
                            ssd_counter += 1
                            image_saved = True;

                        # create ssd entry for person
                        ssd_image_desc = [frame_filename, int(round(obj['xmin'])), int(round(obj['xmax'])), int(round(obj['ymin'])), int(round(obj['ymax'])), 5]
                        ssd_image_list.append(ssd_image_desc)

                    # See https://docs.opencv.org/3.4.1/d6/d6e/group__imgproc__draw.html
                    # for more information about the cv2.rectangle method.
                    # Method signature: image, point1, point2, color, and tickness.
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 165, 20), 10)
                    # Amount to offset the label/probability text above the bounding box.
                    text_offset = 15
                    # See https://docs.opencv.org/3.4.1/d6/d6e/group__imgproc__draw.html
                    # for more information about the cv2.putText method.
                    # Method signature: image, text, origin, font face, font scale, color,
                    # and tickness
                    cv2.putText(frame, "{}: {:.2f}%".format(output_map[obj['label']],
                                                            obj['prob'] * 100),
                                (xmin, ymin-text_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 165, 20), 6)
            # Set the next frame in the local display stream.
            local_display.set_frame_data(frame)

            if image_saved:
                with open('/tmp/cats/train/train.csv', 'a') as outcsv:
                    writer = csv.writer(outcsv)
                    writer.writerows(ssd_image_list)
                    ssd_image_list=[]
                    outcsv.close()

    except Exception as ex:
        client.publish(topic=iot_topic, payload='Error in cat-plus-critter lambda: {}'.format(ex))
        outcsv.close()

catcritter_infinite_infer_run()
