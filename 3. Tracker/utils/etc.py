import os
import random
import tempfile

import numpy as np
import trackeval

# Randomly select bbox color for each object id
color = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(5000)]


def set_parameters(args, vid_name, mode):
    # Set properly for each dataset
    if 'MOT17' in vid_name:
        # Path
        if mode == 'val':
            args.pickle_path = args.pickle_dir + 'mot17_val_0.80.pickle'
            args.pickle_path_95 = args.pickle_dir + 'mot17_val_0.95.pickle'
            args.data_path = args.data_dir + 'MOT17/train/'
        elif mode == 'val_custom':
            args.pickle_path = args.pickle_dir + 'mot17_val_custom_0.80.pickle'
            args.pickle_path_95 = args.pickle_dir + 'mot17_val_custom_0.95.pickle'
            args.data_path = args.data_dir + 'MOT17/train/'
        elif mode == 'train_custom':
            args.pickle_path = args.pickle_dir + 'mot17_train_custom_0.80.pickle'
            args.pickle_path_95 = args.pickle_dir + 'mot17_train_custom_0.95.pickle'
            args.data_path = args.data_dir + 'MOT17/train/'
        elif mode == 'all':
            args.pickle_path = args.pickle_dir + 'mot17_all_0.80.pickle'
            args.pickle_path_95 = args.pickle_dir + 'mot17_all_0.95.pickle'
            args.data_path = args.data_dir + 'MOT17/train/'
        else:
            args.pickle_path = args.pickle_dir + 'mot17_test_0.80.pickle'
            args.pickle_path_95 = args.pickle_dir + 'mot17_test_0.95.pickle'
            args.data_path = args.data_dir + 'MOT17/test/'

        if '01' in vid_name or '03' in vid_name or '12' in vid_name:
            args.det_thr, args.init_thr = 0.65, 0.75
        elif '07' in vid_name:
            args.det_thr, args.init_thr = 0.60, 0.60
        elif '14' in vid_name:
            args.det_thr, args.init_thr = 0.45, 0.55
        else:
            args.det_thr, args.init_thr = 0.60, 0.70
        args.match_thr = 0.70

    elif 'MOT20' in vid_name:
        if mode == 'val':
            args.pickle_path = args.pickle_dir + 'mot20_val_0.80.pickle'
            args.pickle_path_95 = args.pickle_dir + 'mot20_val_0.95.pickle'
            args.data_path = args.data_dir + 'MOT20/train/'
        elif mode == 'val_custom':
            args.pickle_path = args.pickle_dir + 'mot20_val_custom_0.80.pickle'
            args.pickle_path_95 = args.pickle_dir + 'mot20_val_custom_0.95.pickle'
            args.data_path = args.data_dir + 'MOT20/train/'
        elif mode == 'train_custom':
            args.pickle_path = args.pickle_dir + 'mot20_train_custom_0.80.pickle'
            args.pickle_path_95 = args.pickle_dir + 'mot20_train_custom_0.95.pickle'
            args.data_path = args.data_dir + 'MOT20/train/'
        elif mode == 'all':
            args.pickle_path = args.pickle_dir + 'mot20_all_0.80.pickle'
            args.pickle_path_95 = args.pickle_dir + 'mot20_all_0.95.pickle'
            args.data_path = args.data_dir + 'MOT20/train/'
        else:
            args.pickle_path = args.pickle_dir + 'mot20_test_0.80.pickle'
            args.pickle_path_95 = args.pickle_dir + 'mot20_test_0.95.pickle'
            args.data_path = args.data_dir + 'MOT20/test/'

        if '08' in vid_name:
            args.det_thr, args.init_thr = 0.30, 0.40
        if '04' in vid_name or '06' in vid_name or '07' in vid_name:
            args.det_thr, args.init_thr = 0.40, 0.50
        else:
            args.det_thr, args.init_thr = 0.40, 0.40
        args.match_thr = 0.55

    elif 'SportsMOT' in vid_name or 'sportsmot' in vid_name.lower():
        if mode == 'val':
            args.pickle_path = args.pickle_dir + 'sportsmot_val_0.80.pickle'
            args.pickle_path_95 = args.pickle_dir + 'sportsmot_val_0.95.pickle'
            args.data_path = args.data_dir + 'SportsMOT/dataset/val/'
        else:
            args.pickle_path = args.pickle_dir + 'sportsmot_test_0.80.pickle'
            args.pickle_path_95 = args.pickle_dir + 'sportsmot_test_0.95.pickle'
            args.data_path = args.data_dir + 'SportsMOT/dataset/test/'

        # SportsMOT运动场景，需要较高的检测和匹配阈值
        args.det_thr = 0.55
        args.init_thr = 0.65
        args.match_thr = 0.75

    elif 'Dance' in vid_name or 'dancetrack' in vid_name.lower():
        if mode == 'val':
            args.pickle_path = args.pickle_dir + 'dance_val_0.80.pickle'
            args.pickle_path_95 = args.pickle_dir + 'dance_val_0.95.pickle'
            args.data_path = args.data_dir + 'DanceTrack/val/'
        else:
            args.pickle_path = args.pickle_dir + 'dance_test_0.80.pickle'
            args.pickle_path_95 = args.pickle_dir + 'dance_test_0.95.pickle'
            args.data_path = args.data_dir + 'DanceTrack/test/'

        # Baseline Setting
        args.det_thr = 0.60
        args.init_thr = 0.60
        args.match_thr = 0.80 if mode == 'val' else 0.60


def write_results(filename, results):
    # Set save format
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'

    # Open file
    f = open(filename, 'w')

    # Write
    for frame_id, track_ids, x1y1whs, scores in results:
        for track_id, x1y1wh, score in zip(track_ids, x1y1whs, scores):
            # Get box
            x1, y1, w, h = x1y1wh

            # Generate line to write
            line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1),
                                      w=round(w, 1), h=round(h, 1), s=round(score, 2))

            # Write
            f.write(line)

    # Close
    f.close()


def _default_trackeval_log_path():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'error_log.txt')
    )


def _trackeval_seqmap_file(seqmap_folder, split_name):
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'trackeval', 'seqmap', seqmap_folder, f'{split_name}.txt')
    )


def _get_trackeval_benchmark_and_seqmap(dataset):
    lowered = dataset.lower()
    if 'dancetrack' in lowered:
        return 'MOT17', 'dancetrack'
    if 'sportsmot' in lowered:
        return 'MOT17', 'sportsmot'
    if 'mot20' in lowered:
        return 'MOT20', 'mot20'
    return 'MOT17', 'mot17'


def get_trackeval_configs(
    args,
    trackers_to_eval,
    dataset,
    output_folder=None,
    output_summary=False,
    output_detailed=False,
    tracker_sub_folder='',
    output_sub_folder='',
):
    # Determine split name for seqmap
    split_name = args.mode if args.mode in ['val', 'val_custom', 'train_custom', 'all'] else 'val'

    benchmark, seqmap_folder = _get_trackeval_benchmark_and_seqmap(dataset)
    if isinstance(trackers_to_eval, str):
        trackers = [trackers_to_eval]
    else:
        trackers = list(trackers_to_eval)

    # Set evaluation configurations
    eval_config = {'USE_PARALLEL': False,
                   'NUM_PARALLEL_CORES': 1,
                   'BREAK_ON_ERROR': True,
                   'RETURN_ON_ERROR': False,
                   'LOG_ON_ERROR': _default_trackeval_log_path(),

                   'PRINT_RESULTS': False,
                   'PRINT_ONLY_COMBINED': False,
                   'PRINT_CONFIG': False,
                   'TIME_PROGRESS': False,
                   'DISPLAY_LESS_PROGRESS': True,

                   'OUTPUT_SUMMARY': output_summary,
                   'OUTPUT_EMPTY_CLASSES': output_summary or output_detailed,
                   'OUTPUT_DETAILED': output_detailed,
                   'PLOT_CURVES': False}

    dataset_config = {'GT_FOLDER': args.data_path,
                      'TRACKERS_FOLDER': args.output_dir,
                      'OUTPUT_FOLDER': output_folder,
                      'TRACKERS_TO_EVAL': trackers,
                      'CLASSES_TO_EVAL': ['pedestrian'],
                      'BENCHMARK': benchmark,
                      'SPLIT_TO_EVAL': split_name,
                      'INPUT_AS_ZIP': False,
                      'PRINT_CONFIG': False,
                      'DO_PREPROC': True,
                      'TRACKER_SUB_FOLDER': tracker_sub_folder,
                      'OUTPUT_SUB_FOLDER': output_sub_folder,
                      'TRACKER_DISPLAY_NAMES': None,
                      'SEQMAP_FOLDER': None,
                      'SEQMAP_FILE': _trackeval_seqmap_file(seqmap_folder, split_name),
                      'SEQ_INFO': None,
                      'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',
                      'SKIP_SPLIT_FOL': True}
    return eval_config, dataset_config


def evaluate(args, trackers_to_eval, dataset):
    eval_config, dataset_config = get_trackeval_configs(args, trackers_to_eval, dataset)

    # Set configuration
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    res, _ = evaluator.evaluate(dataset_list, metrics_list)

    # Get
    hota = np.mean(res['MotChallenge2DBox'][trackers_to_eval]['COMBINED_SEQ']['pedestrian']['HOTA']['HOTA']).item()
    idf1 = res['MotChallenge2DBox'][trackers_to_eval]['COMBINED_SEQ']['pedestrian']['Identity']['IDF1']
    mota = res['MotChallenge2DBox'][trackers_to_eval]['COMBINED_SEQ']['pedestrian']['CLEAR']['MOTA']
    assa = np.mean(res['MotChallenge2DBox'][trackers_to_eval]['COMBINED_SEQ']['pedestrian']['HOTA']['AssA']).item()
    deta = np.mean(res['MotChallenge2DBox'][trackers_to_eval]['COMBINED_SEQ']['pedestrian']['HOTA']['DetA']).item()

    # Print
    print(f'{"HOTA":<10}{"MOTA":<10}{"IDF1":<10}{"DetA":<10}{"AssA":<10}', flush=True)
    print(f'{hota:<10.6f}{mota:<10.6f}{idf1:<10.6f}{deta:<10.6f}{assa:<10.6f}', flush=True)
    if getattr(args, 'print_per_sequence_metrics', False):
        print_per_sequence_metrics(res, trackers_to_eval)


def print_per_sequence_metrics(res, trackers_to_eval):
    """Print per-sequence HOTA/CLEAR/Identity metrics without changing combined parsing."""
    tracker_res = res['MotChallenge2DBox'][trackers_to_eval]
    sequence_names = [seq for seq in tracker_res.keys() if seq != 'COMBINED_SEQ']
    if not sequence_names:
        return

    print('Per-sequence metrics:', flush=True)
    print(f'{"SEQ":<24}{"HOTA":<10}{"MOTA":<10}{"IDF1":<10}{"DetA":<10}{"AssA":<10}', flush=True)
    for seq in sequence_names:
        seq_res = tracker_res[seq]['pedestrian']
        seq_hota = np.mean(seq_res['HOTA']['HOTA']).item()
        seq_idf1 = seq_res['Identity']['IDF1']
        seq_mota = seq_res['CLEAR']['MOTA']
        seq_assa = np.mean(seq_res['HOTA']['AssA']).item()
        seq_deta = np.mean(seq_res['HOTA']['DetA']).item()
        print(
            f'{seq:<24}{seq_hota:<10.6f}{seq_mota:<10.6f}'
            f'{seq_idf1:<10.6f}{seq_deta:<10.6f}{seq_assa:<10.6f}',
            flush=True,
        )
    print('', flush=True)


def evaluate_sequences(args, trackers_to_eval, dataset, sequences):
    """Evaluate an existing tracker folder on a custom subset of sequences."""
    eval_config, dataset_config = get_trackeval_configs(args, trackers_to_eval, dataset)

    seqmap_tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, prefix='seqmap_')
    try:
        seqmap_tmp.write('name\n')
        for seq in sequences:
            seqmap_tmp.write(seq + '\n')
        seqmap_tmp.close()
        dataset_config['SEQMAP_FILE'] = seqmap_tmp.name

        evaluator = trackeval.Evaluator(eval_config)
        dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
        metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
        res, _ = evaluator.evaluate(dataset_list, metrics_list)

        hota = np.mean(res['MotChallenge2DBox'][trackers_to_eval]['COMBINED_SEQ']['pedestrian']['HOTA']['HOTA']).item()
        idf1 = res['MotChallenge2DBox'][trackers_to_eval]['COMBINED_SEQ']['pedestrian']['Identity']['IDF1']
        mota = res['MotChallenge2DBox'][trackers_to_eval]['COMBINED_SEQ']['pedestrian']['CLEAR']['MOTA']
        assa = np.mean(res['MotChallenge2DBox'][trackers_to_eval]['COMBINED_SEQ']['pedestrian']['HOTA']['AssA']).item()
        deta = np.mean(res['MotChallenge2DBox'][trackers_to_eval]['COMBINED_SEQ']['pedestrian']['HOTA']['DetA']).item()

        print(f'{"HOTA":<10}{"MOTA":<10}{"IDF1":<10}{"DetA":<10}{"AssA":<10}', flush=True)
        print(f'{hota:<10.6f}{mota:<10.6f}{idf1:<10.6f}{deta:<10.6f}{assa:<10.6f}', flush=True)
        if getattr(args, 'print_per_sequence_metrics', False):
            print_per_sequence_metrics(res, trackers_to_eval)
    finally:
        try:
            os.unlink(seqmap_tmp.name)
        except FileNotFoundError:
            pass
