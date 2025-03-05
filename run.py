import os
import argparse
import time
import torch
import numpy as np
import pandas as pd
import onnxruntime as ort
from torchvision import transforms
from torch.utils.data import DataLoader
from src.utils.vision import Dataset, ort_yolo_predict, ort_eb_car_predict, non_max_suppression, extract_object_tensor_pairs, resize_tensor_pairs, scale_bboxes
from src.utils.vision import prepare_data_batch, get_ort_face_function
from src.utils.logging import store_labels, query_yes_no


def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=False, default='models',
                        dest='model_dir', help='Directory where models are stored')
    parser.add_argument('--data_dir', required=False, default='data/preprocessed_2fps',
                        dest='data_dir', help='Directory to read from')
    parser.add_argument('--output_dir', required=False, default='profiling/output',
                        dest='output_dir', help='Directory to write output logs in')
    parser.add_argument('--profiling_dir', required=False, default='profiling',
                        dest='profiling_dir', help=f'Directory to write profiled '
                        f'data in')
    parser.add_argument('--obj_det_model', '-od', required=True, dest='obj_det_model',
                        help=f'Object detection model to use. Options are: yolov5n, '
                        f'yolov5s, yolov5m, yolov5l, yolov5x',
                        choices=['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'])
    parser.add_argument('--face_rec_model', '-fr', required=False, dest='face_rec_model',
                        default=None, help=f'Face recognition model to use. If none is '
                        f'specified, branch will not be used. Options are: resnet50, '
                        f'senet50, vgg16', choices=['resnet50', 'senet50', 'vgg16',
                                                    'genderNet', 'genderNet_11',
                                                    'genderNet_16', 'genderNet_19'])
    parser.add_argument('--car_cls_model', '-cc', required=False, dest='car_cls_model',
                        default=None, help=f'Car classification model to use. If none is '
                        f'specified, branch will not be used. Options are: eb0, eb1, eb2, '
                        f'eb3, eb4, eb5, eb6, eb7', choices=['eb0', 'eb1', 'eb2', 'eb3',
                                                             'eb4', 'eb5', 'eb6', 'eb7'])
    parser.add_argument('--obj_det_batch_size', '-odb', required=False, default=16,
                        dest='obj_det_batch_size', help=f'Batch size for object '
                        f'detection model. Default is 16')
    parser.add_argument('--save_intermediate', '-s', required=False, default=False,
                        dest='save_intermediate', action='store_true',
                        help='Save intermediate cropped images')
    parser.add_argument('--num_images', required=False, default='-1',
                        dest='num_images', help='Maximum number of images to use')
    return parser.parse_args()

def main(args):
    PERSON_LABEL = 0
    CAR_LABEL = 2

    od_batch_size = int(args.obj_det_batch_size)

    # Profiling overhead is not measured in the pipeline currently
    BRANCH_FREQUENCY_PROFILING = True
    STORE_ALL_LABELS = True

    # Certain branches can be shut off for debugging or performance purposes
    if args.face_rec_model is None:
        PERSON_BRANCH_ON = False
    else:
        PERSON_BRANCH_ON = True
    if args.car_cls_model is None:
        CAR_BRANCH_ON = False
    else:
        CAR_BRANCH_ON = True

    num_images = int(args.num_images)

    # Whether to save intermediate results or not
    save_intermediate = args.save_intermediate
    last_person_idx = 0
    last_car_idx = 0
    
    # model_base_dir = '/work/pi_rsitaram_umass_edu/sohaib/models/'
    model_dir = args.model_dir
    data_dir = args.data_dir
    profiling_dir = args.profiling_dir
    output_dir = args.output_dir

    if not(os.path.exists(model_dir)):
        print(f'Model directory does not exist: {model_dir}\nExiting..')
        exit(1)
    if not(os.path.exists(data_dir)):
        print(f'Data directory does not exist: {data_dir}\nExiting..')
        exit(1)
    if not(os.path.exists(profiling_dir)):
        createDir = query_yes_no(f'Profiling directory does not exist: '
                                 f'{profiling_dir}\nCreate directory and proceed?')
        if createDir:
            os.makedirs(profiling_dir)
            print(f'Created profiling directory: {output_dir}')
        else:
            print(f'Directory not created, exiting..')
            exit(1)
    if not(os.path.exists(output_dir)):
        createDir = query_yes_no(f'Output directory does not exist: {output_dir}'
                                 f'\nCreate directory and proceed?')
        if createDir:
            os.makedirs(output_dir)
            print(f'Created output directory: {output_dir}')
        else:
            print(f'Directory not created, exiting..')
            exit(1)

    # data_dir = '/work/pi_rsitaram_umass_edu/sohaib/datasets/traffic_nexus/4_Olive_NS/preprocessed/'
    # data_dir = '/work/pi_rsitaram_umass_edu/sohaib/datasets/bellevue_traffic/preprocessed/'
    frames_per_sec = 2
    dataset = Dataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=od_batch_size, shuffle=False)

    # models = ['yolv5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
    obj_det_model_name = args.obj_det_model
    obj_det_model_path = os.path.join(model_dir, obj_det_model_name+'.onnx')
    obj_det_model = ort.InferenceSession(obj_det_model_path,
                                         providers=['CUDAExecutionProvider'])
    # obj_det_model = ort.InferenceSession(obj_det_model_path, providers=['TensorrtExecutionProvider'])
    # print(obj_det_model)

    if PERSON_BRANCH_ON:
        face_rec_model_name = args.face_rec_model
        if 'genderNet' in face_rec_model_name:
            # face_rec_model_path = (f'/work/pi_rsitaram_umass_edu/sohaib/facial_'
            #                        f'age_gender/{face_rec_model_name}_best.onnx')
            face_rec_model_path = os.path.join(model_dir, face_rec_model_name+'_best.onnx')
        else:
            face_rec_model_name = face_rec_model_name + '_face'
            face_rec_model_path = os.path.join(model_dir, face_rec_model_name+'.onnx')
        face_rec_model = ort.InferenceSession(face_rec_model_path,
                                              providers=['CUDAExecutionProvider'])
        ort_face_predict = get_ort_face_function(face_rec_model_name)
        # face_rec_model = ort.InferenceSession(face_rec_model_path,
        #                                       providers=['TensorrtExecutionProvider'])

    if CAR_BRANCH_ON:
        car_cls_model_name = f'{args.car_cls_model}_checkpoint_150epochs'
        car_cls_model_path = os.path.join(model_dir, car_cls_model_name+'.onnx')
        car_cls_model = ort.InferenceSession(car_cls_model_path,
                                             providers=['CUDAExecutionProvider'])
        # car_cls_model = ort.InferenceSession(car_cls_model_path,
        #                                      providers=['TensorrtExecutionProvider'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    image_id = 0
    start_time = time.time()
    obj_det_pre_times = []
    obj_det_inf_times = []
    obj_det_post_times = []
    face_rec_pre_times = []
    face_rec_inf_times = []
    face_rec_post_times = []
    car_cls_pre_times = []
    car_cls_inf_times = []
    car_cls_post_times = []
    pipeline_times = []

    face_branch_items = []
    car_branch_items = []

    if STORE_ALL_LABELS:
        all_obj_det_tensors = torch.tensor(np.zeros((1, 8)), device=device)
        face_rec_tensors = torch.tensor(np.zeros((1, 10)), device=device)
        car_cls_tensors = torch.tensor(np.zeros((1, 10)), device=device)


    exp_start_time = time.time()
    for data in dataloader:
    # for data in dataset:
        print(f'data.shape: {data.shape}')

        pipeline_start_time = time.time()
        # ------------------------------------
        # Pipeline step: Object detection pre-processing
        # ------------------------------------
        start_time = time.time()
        # TODO: remove hard-coded values
        # we use 1280x1280 to profile accuracy better, but it is very slow
        # in real-time we can use 640x640 to meet SLOs
        obj_det_input_size = [1280, 1280]
        resized_data = transforms.Resize(size=obj_det_input_size)(data)
        pre_time = time.time() - start_time
        obj_det_pre_times.append(pre_time)
        # ------------------------------------

        # ------------------------------------
        # Pipeline step: Object detection
        # ------------------------------------
        inf_start_time = time.time()
        outputs = ort_yolo_predict(obj_det_model, resized_data)
        inf_end_time = time.time()
        inf_time = inf_end_time - inf_start_time
        obj_det_inf_times.append(inf_time)

        print(f'DEBUG: Object detection inference time: {inf_time}')
        print(f'DEBUG: Object detection inference time per image: {inf_time/data.shape[0]}')

        # Now that we have executed a batch of object detection, treat each image
        # separately since we can't mix labels from different images
        batch_img_counter = 0
        print(f'outputs: {outputs}')
        outputs = outputs['ort_outs']
        for output in outputs[0]:
            single_output = np.expand_dims(output, axis=0)
            # TODO: data copying overhead could be reduced, but seems like it is
            #       not practically making any difference. Perhaps data is already
            #       on GPU?
            start_time = time.time()
            nms_pred = non_max_suppression(torch.tensor(np.array(single_output),
                                                        device=device))
            scaled_objdet_pred = scale_bboxes(preds=nms_pred, original_img=data,
                                            downscaled_size=obj_det_input_size)
            
            scaled_objdet_tensors = torch.tensor(np.zeros((1, 8)), device=device)

            if STORE_ALL_LABELS:
                for _pred in scaled_objdet_pred:
                    # for every image in the batch
                    for det in _pred:
                        # for every object detected in the image
                        timestamp = image_id / frames_per_sec
                        concat_tensor = torch.tensor([image_id, det[5], det[4],
                                                      det[0], det[1], det[2],
                                                      det[3], timestamp],
                                                    device=device)
                        concat_tensor = concat_tensor[None, :]
                        scaled_objdet_tensors = torch.cat((scaled_objdet_tensors,
                                                           concat_tensor))
                scaled_objdet_tensors = scaled_objdet_tensors[1:, :]
                all_obj_det_tensors = torch.cat((all_obj_det_tensors,
                                                 scaled_objdet_tensors))
            post_time = time.time() - start_time
            obj_det_post_times.append(post_time)
            # ------------------------------------

            # TODO: move all pre-processing and pre+post-processing under a function
            #       a function called object_detection (or similar name) and
            #       move non-max suppression and pre+post-processing inside it
            
            # ------------------------------------
            # Pipeline step: Cropping and resizing
            # ------------------------------------

            # Person branch
            start_time = time.time()
            person_tensor_pairs = extract_object_tensor_pairs(img_tensor=data[batch_img_counter],
                                                                obj_det_tensors=scaled_objdet_tensors,
                                                                label=PERSON_LABEL)
            
            if BRANCH_FREQUENCY_PROFILING:
                face_branch_items.append(len(person_tensor_pairs))

            if PERSON_BRANCH_ON:
                # TODO: remove hard-coded value. But also, it should not be confused by
                #       [3,224,224] or [224,224,3] etc across different models
                # face_rec_input_size = face_rec_model.get_inputs()[0].shape[1:3]
                face_rec_input_size = [224, 224]
                person_tensor_pairs = resize_tensor_pairs(person_tensor_pairs,
                                                            face_rec_input_size)

                if save_intermediate:
                    person_tensors = list(map(lambda x: x[0], person_tensor_pairs))
                    last_person_idx = save_images(person_tensors,
                                                    f'intermediate/{obj_det_model_name}',
                                                    'person', last_person_idx)
                    
            pre_time = time.time() - start_time
            face_rec_pre_times.append(pre_time)
            
            # Car branch
            start_time = time.time()
            car_tensor_pairs = extract_object_tensor_pairs(img_tensor=data[batch_img_counter],
                                                            obj_det_tensors=scaled_objdet_tensors,
                                                            label=CAR_LABEL)
            
            if BRANCH_FREQUENCY_PROFILING:
                car_branch_items.append(len(car_tensor_pairs))

            if CAR_BRANCH_ON:
                car_cls_input_size = car_cls_model.get_inputs()[0].shape[-2:]
                # print(f'car_tensors len: {len(car_tensors)}')
                # print(f'car_tensors[0] shape: {car_tensors[0].shape}')
                # print(f'car_cls_model input shape: {car_cls_model.get_inputs()[0].shape}')
                car_tensor_pairs = resize_tensor_pairs(car_tensor_pairs, car_cls_input_size)

                if save_intermediate:
                    car_tensors = list(map(lambda x: x[0], car_tensor_pairs))
                    last_car_idx = save_images(car_tensors,
                                                f'intermediate/{obj_det_model_name}',
                                                'car', last_car_idx)
            pre_time = time.time() - start_time
            car_cls_pre_times.append(pre_time)
            # ------------------------------------

            # ------------------------------------
            # Pipeline step: Facial recognition
            # ------------------------------------
            if PERSON_BRANCH_ON:
                # TODO: we create a batch but have not used it yet
                # TODO: the batch is created from all the persons detected in a SINGLE image
                person_tensors = list(map(lambda x: x[0], person_tensor_pairs))
                batched_tensors = prepare_data_batch(person_tensors)
                if batched_tensors is not None:
                    print(f'batched_tensors.shape: {batched_tensors.shape}')
                for person_tensor_pair in person_tensor_pairs:
                    # ------------------------------------
                    # Pipeline sub-set: Inference
                    # ------------------------------------
                    start_time = time.time()
                    person_tensor = person_tensor_pair[0]
                    obj_det_tensor = person_tensor_pair[1]
                    face_output = ort_face_predict(face_rec_model, person_tensor)
                    inf_time = time.time() - start_time
                    face_rec_inf_times.append(inf_time)

                    print(f'DEBUG: Person branch inference time: {inf_time}')

                    # ------------------------------------
                    # Pipeline sub-set: Post-processing
                    # ------------------------------------
                    start_time = time.time()
                    face_label = np.argmax(face_output[0])
                    face_score = np.max(face_output[0])

                    # print(f'face_label: {face_label}, face_score: {face_score}, '
                    #       f'face_output: {face_output}')

                    if STORE_ALL_LABELS:
                        timestamp = image_id / frames_per_sec
                        pre_concat_tensor = torch.tensor([image_id, face_label,
                                                          face_score],
                                                        device=device)
                        # The [None, :] is needed to convert 1D tensor to 2D tensor
                        concat_tensor = torch.cat((pre_concat_tensor[None, :],
                                                obj_det_tensor[1:][None, :]), dim=1)
                        face_rec_tensors = torch.cat((face_rec_tensors, concat_tensor))

                    post_time = time.time() - start_time
                    face_rec_post_times.append(post_time)
            # ------------------------------------

            # ------------------------------------
            # Pipeline step: Car classification
            # ------------------------------------
            if CAR_BRANCH_ON:
                for car_tensor_pair in car_tensor_pairs:
                    # ------------------------------------
                    # Pipeline sub-set: Inference
                    # ------------------------------------
                    start_time = time.time()
                    car_tensor = car_tensor_pair[0]
                    obj_det_tensor = car_tensor_pair[1]
                    car_output = ort_eb_car_predict(car_cls_model, car_tensor)
                    inf_time = time.time() - start_time
                    car_cls_inf_times.append(inf_time)

                    print(f'DEBUG: Car branch inference time: {inf_time}')

                    # ------------------------------------
                    # Pipeline sub-set: Post-processing
                    # ------------------------------------
                    start_time = time.time()
                    car_label = np.argmax(car_output[0])
                    car_score = np.max(car_output[0])

                    if STORE_ALL_LABELS:
                        timestamp = image_id / frames_per_sec
                        pre_concat_tensor = torch.tensor([image_id, car_label,
                                                          car_score],
                                                        device=device)
                        # The [None, :] is needed to convert 1D tensor to 2D tensor
                        concat_tensor = torch.cat((pre_concat_tensor[None, :],
                                                obj_det_tensor[1:][None, :]), 1)
                        car_cls_tensors = torch.cat((car_cls_tensors, concat_tensor))

                    post_time = time.time() - start_time
                    car_cls_post_times.append(post_time)
                batch_img_counter += 1
            # ------------------------------------

        pipeline_time = time.time() - pipeline_start_time
        pipeline_times.append(pipeline_time)

        image_id += od_batch_size
        if num_images > 0 and image_id >= num_images:
            break

    total_time = time.time() - exp_start_time
    print()
    print(f'Total time for {num_images} images: {total_time}')
    print()
    print(f'Object detection pre-processing median time: {np.median(obj_det_pre_times)}')
    # print(f'List: {obj_det_pre_times}')
    print(f'Object detection inference median time: {np.median(obj_det_inf_times)}')
    # print(f'List: {obj_det_inf_times}')
    print(f'Object detection post-processing median time: {np.median(obj_det_post_times)}')
    # print(f'List: {obj_det_post_times}')
    print()
    print(f'Facial recognition pre-processing median time: {np.median(face_rec_pre_times)}')
    # print(f'List: {face_rec_pre_times}')
    print(f'Facial recognition inference median time: {np.median(face_rec_inf_times)}')
    # print(f'List: {face_rec_inf_times}')
    print(f'Facial recognition post-processing median time: {np.median(face_rec_post_times)}')
    # print(f'List: {face_rec_post_times}')
    print()
    print(f'Car classification pre-processing median time: {np.median(car_cls_pre_times)}')
    # print(f'List: {car_cls_pre_times}')
    print(f'Car classification inference median time: {np.median(car_cls_inf_times)}')
    # print(f'List: {car_cls_inf_times}')
    print(f'Car classification post-processing median time: {np.median(car_cls_post_times)}')
    # print(f'List: {car_cls_post_times}')
    print()
    print(f'End-to-end pipeline median time: {np.median(pipeline_times)}')
    print(f'End-to-end pipeline 90th percentile time: {np.percentile(pipeline_times, 90)}')
    print()

    try:
        existing_df = pd.read_csv(os.path.join(profiling_dir, 'runtimes.csv'))
    except FileNotFoundError:
        profiled_dict = {'model': [], 'accelerator': [], 'batch_size': [],
                         'median_runtimes': [], 'max_runtimes': [],
                         'min_runtimes': [], 'avg_runtimes': [],
                         'median_preprocessing_times': [],
                         'median_postprocessing_times': [],
                         'avg_preprocessing_times': [],
                         'avg_postprocessing_times': []}
        existing_df = pd.DataFrame(data=profiled_dict)


    # TODO: remove hard-coded values, fix issue with duplicate rows
    def update_existing_df(existing_df, model_name, inf_times, preprocessing_times,
                        postprocessing_times, accelerator='gtx_1080ti', batch_size=1):
        existing_df = existing_df[~((existing_df['model'] == model_name) & \
                                    (existing_df['accelerator'] == accelerator) & \
                                    (existing_df['batch_size'] == str(batch_size)))]
        # print(existing_df['model'] == model_name)
        # print(existing_df['accelerator'] == accelerator)
        # print(existing_df['batch_size'] == str(batch_size))
        # print()
        # print(f'model: {model_name}, batch_size: {int(batch_size)}')
        new_data_dict = {'model': [model_name],
                        'accelerator': [accelerator],
                        'batch_size': [str(batch_size)],
                        'median_runtimes': [np.median(inf_times)],
                        'max_runtimes': [max(inf_times)],
                        'min_runtimes': [min(inf_times)],
                        'avg_runtimes': [np.average(inf_times)],
                        'median_preprocessing_times': [np.median(preprocessing_times)],
                        'median_postprocessing_times': [np.median(postprocessing_times)],
                        'avg_preprocessing_times': [np.average(preprocessing_times)],
                        'avg_postprocessing_times': [np.average(postprocessing_times)]
                        }
        new_df = pd.DataFrame(data=new_data_dict)
        existing_df = pd.concat([existing_df, new_df], ignore_index=True)

        # existing_df = existing_df[~(existing_df['model'] == 'empty')]
        existing_df = existing_df.loc[:, ~existing_df.columns.str.contains('^Unnamed')]

        return existing_df

    existing_df = update_existing_df(existing_df, obj_det_model_name,
                                     obj_det_inf_times, obj_det_pre_times,
                                     obj_det_post_times)
    if PERSON_BRANCH_ON:
        existing_df = update_existing_df(existing_df, face_rec_model_name,
                                         face_rec_inf_times, face_rec_pre_times,
                                         face_rec_post_times)
    if CAR_BRANCH_ON:
        existing_df = update_existing_df(existing_df, car_cls_model_name,
                                         car_cls_inf_times, car_cls_pre_times,
                                         car_cls_post_times)

    existing_df.to_csv(os.path.join(profiling_dir, 'runtimes.csv'))

    dataset_name = data_dir.split('/')[-1]

    frames = range(len(face_branch_items))
    df = pd.DataFrame(data = {'frame': frames,
                              'car_classification': car_branch_items,
                              'facial_recognition': face_branch_items})
    df.to_csv(os.path.join(output_dir, f'branching_{obj_det_model_name}_{dataset_name}.csv'))


    if STORE_ALL_LABELS:
        store_labels(tensors=all_obj_det_tensors,
                    columns=['frame', 'class', 'score', 'xmin', 'ymin', 'xmax',
                             'ymax', 'timestamp'],
                    output_dir=output_dir,
                    filename=f'{obj_det_model_name}_{dataset_name}.csv')
        
        if CAR_BRANCH_ON:
            store_labels(tensors=car_cls_tensors,
                        columns=['frame', 'class', 'score', 'obj_class',
                                 'obj_score', 'xmin', 'ymin', 'xmax', 'ymax',
                                 'timestamp'],
                        output_dir=output_dir,
                        filename=f'{car_cls_model_name}_{obj_det_model_name}.csv')

        if PERSON_BRANCH_ON:
            store_labels(tensors=face_rec_tensors,
                        columns=['frame', 'class', 'score', 'obj_class',
                                 'obj_score', 'xmin', 'ymin', 'xmax', 'ymax',
                                 'timestamp'],
                        output_dir=output_dir,
                        filename=f'{face_rec_model_name}_{obj_det_model_name}.csv')   


if __name__=='__main__':
    main(getargs())
