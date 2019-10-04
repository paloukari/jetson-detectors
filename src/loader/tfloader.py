import sys
sys.path.insert(1, '/')
from tf_trt_models.detection import download_detection_model
from tf_trt_models.detection import build_detection_graph

def load_model(model_name):
    
    # Download and load the model
    config_path, checkpoint_path = download_detection_model(
        model_name, './models/')

    frozen_graph, input_names, output_names = build_detection_graph(
        config=config_path,
        checkpoint=checkpoint_path
    )

    print(f'Converting {model_name} to trt..')
    trt_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph,
        outputs=output_names,
        max_batch_size=1,
        max_workspace_size_bytes=1 << 25,
        precision_mode='FP16',
        minimum_segment_size=50
    )
    with open(trt_output_file, 'wb') as f:
        f.write(trt_graph.SerializeToString())
        print(f'{trt_output_file} saved.')

    print(f'Model {model_name} converted.')

    print(f'Input names: {input_names}')
    print(f'Output names: {output_names}')

    return trt_graph