import sys
import tensorflow.contrib.tensorrt as trt

sys.path.insert(1, '/')
from tf_trt_models.detection import download_detection_model
from tf_trt_models.detection import build_detection_graph

def load_model(model_name):
    
    # Download and load the model
    config_path, checkpoint_path = download_detection_model(
        model_name, './models/')

    tr_graph, input_names, output_names = build_detection_graph(
        config=config_path,
        checkpoint=checkpoint_path
    )

    print(f'Input names: {input_names}')
    print(f'Output names: {output_names}')

    return tr_graph