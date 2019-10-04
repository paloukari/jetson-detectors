import time
import threading
import cv2

class ObjectDetectionAsync:
    def __init__(self, model_name, trt_optimize, frame_callback):
        self.started = False
        self.model_name = model_name
        self.trt_optimize = trt_optimize 
        self.frame_callback = frame_callback

        self.scores = [0]
        self.boxes = [0]
        self.classes = [0]
        self.num_detections = [0]

        self._ready = False

    def start(self):
        if self.started:
            print('[!] Asynchronous detector already started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self
    
    def _create(self):
        import tensorflow as tf
        from loader.tfloader import load_model as tf_loader
        from loader.trtloader import load_model as trt_loader

        trt_graph = None
        if self.trt_optimize:
            trt_graph = trt_loader(self.model_name)
        else:
            trt_graph = tf_loader(self.model_name)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.tf_sess = tf.Session(config=tf_config)
        tf.import_graph_def(trt_graph, name='')

        self.tf_input = self.tf_sess.graph.get_tensor_by_name('image_tensor:0')
        self.tf_scores = self.tf_sess.graph.get_tensor_by_name('detection_scores:0')
        self.tf_boxes = self.tf_sess.graph.get_tensor_by_name('detection_boxes:0')
        self.tf_classes = self.tf_sess.graph.get_tensor_by_name('detection_classes:0')
        self.tf_num_detections = self.tf_sess.graph.get_tensor_by_name('num_detections:0')
        
        self.frame =  None
        # wait for the video feed to activate
        while self.frame is None:
             time.sleep(0.1)
             self.frame = self.frame_callback()
             
        # run once to load all cuda libs
        self.scores, self.boxes, self.classes, self.num_detections = self.tf_sess.run(
                        [self.tf_scores, self.tf_boxes, self.tf_classes, self.tf_num_detections], feed_dict={self.tf_input: self.frame[None, ...]})
        self._ready = True

    def update(self):
        self._create()
        while self._ready:
            self.frame = self.frame_callback()

            if self.frame is not None:
                try:
                    self.scores, self.boxes, self.classes, self.num_detections = self.tf_sess.run(
                        [self.tf_scores, self.tf_boxes, self.tf_classes, self.tf_num_detections], feed_dict={self.tf_input: self.frame[None, ...]})
                except Exception as ex:
                    print(ex)
                
    @property
    def ready(self):
        return self._ready

    def read(self):
        self.scores[0], self.boxes[0], self.classes[0], self.num_detections[0]

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()
