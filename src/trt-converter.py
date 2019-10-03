import sys
sys.path.insert(1, '/')
from tf_trt_models.detection import download_detection_model
from tf_trt_models.detection import build_detection_graph
from tensorflow.python.compiler.tensorrt import trt_convert as trt


def main(argv):

    model_name = 'ssd_inception_v2_coco'

    if len(argv) == 1:
        model_name = argv[0]

    trt_output_file = f'./models/{model_name}_trt.pb'

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


if __name__ == "__main__":
    main(sys.argv[1:])
