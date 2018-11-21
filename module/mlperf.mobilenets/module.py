#
# Collective Knowledge (Raw data access (json))
#
#
#
#
# Developer:
#

cfg={}  # Will be updated by CK (meta description of this module)
work={} # Will be updated by CK (temporal data)
ck=None # Will be updated by CK (initialized CK kernel)

import os
import sys
import json
import re

import pandas as pd
import numpy as np

# Local settings

##############################################################################
# Initialize module

def init(i):
    """

    Input:  {}

    Output: {
              return       - return code =  0, if successful
                                         >  0, if error
              (error)      - error text if return > 0
            }

    """
    return {'return':0}

##############################################################################
# get raw data for repo widget

def get_raw_data(i):
    """
    Input:  {
              selected_repo     - which repository to take the data from.
                                    If explicitly set to '' will not filter by repository and take all available data.
            }

    Output: {
              return       - return code =  0, if successful
                                         >  0, if error
              (error)      - error text if return > 0
            }

    """

    def get_experimental_results(repo_uoa, tags='explore-mobilenets-accuracy', accuracy=True,
                                 module_uoa='experiment', _library=None, _platform=None):

        r = ck.access({'action':'search', 'repo_uoa':repo_uoa, 'module_uoa':module_uoa, 'tags':tags})
        if r['return']>0:
            print('Error: %s' % r['error'])
            exit(1)
        experiments = r['lst']

        dfs = []
        for experiment in experiments:
            data_uoa = experiment['data_uoa']
            r = ck.access({'action':'list_points', 'repo_uoa':repo_uoa, 'module_uoa':module_uoa, 'data_uoa':data_uoa})
            if r['return']>0:
                print('Error: %s' % r['error'])
                exit(1)
            # Mapping of expected library tags to reader-friendly names.
            tag_to_name = {
                # ArmCL tags on HiKey.
                '17.12-48bc34ea'    : 'armcl-17.12',
                '18.01-f45d5a9b'    : 'armcl-18.01',
                '18.03-e40997bb'    : 'armcl-18.03',
                'request-d8f69c13'  : 'armcl-dv/dt', # armcl-18.03+
                '18.05-b3a371bc'    : 'armcl-18.05',
                '18.08-52ba29e9'    : 'armcl-18.08',
                # ArmCL tags on Firefly.
                '17.12-48bc34e'     : 'armcl-17.12',
                '18.01-f45d5a9'     : 'armcl-18.01',
                '18.03-e40997b'     : 'armcl-18.03',
                '18.05-b3a371b'     : 'armcl-18.05',
                '18.08-52ba29e'     : 'armcl-18.08',
                # TensorFlow tags.
                'tensorflow-1.7'    : 'tensorflow-1.7',
                'tensorflow-1.8'    : 'tensorflow-1.8',
                'tflite-0.1.7'      : 'tflite-0.1.7',
            }

            # TODO: Move the platform mappings to meta.
            # Linaro HiKey960
            hikey_model = 'HiKey960\x00'
            hikey_name  = 'Linaro HiKey960'
            hikey_id    = 'hikey-960'
            hikey_gpu   = 'Mali-G71 MP8'
            hikey_gpu_mhz = '807 MHz'
            # Firefly RK3399
            firefly_model = 'Rockchip RK3399 Firefly Board (Linux Opensource)\x00'
            firefly_name  = 'Firefly RK3399'
            firefly_id    = 'firefly'
            firefly_gpu   = 'Mali-T860 MP4'
            firefly_gpu_mhz = '800 MHz'
            # Huawei Mate 10 Pro
            mate_model      = 'BLA-L09'
            mate_name       = 'Huawei BLA-L09'
            mate_id         = 'mate'
            mate_gpu        = 'Mali-G72 MP12'
            mate_gpu_mhz    = '767 MHz'
            # Platform mappings
            model_to_id = {
                firefly_model : firefly_id,
                hikey_model   : hikey_id,
                mate_model    : mate_id,
            }
            id_to_name = {
                firefly_id : firefly_name,
                hikey_id   : hikey_name,
                mate_id    : mate_name,
            }
            id_to_gpu = {
                firefly_id : firefly_gpu,
                hikey_id   : hikey_gpu,
                mate_id    : mate_gpu,
            }
            id_to_gpu_mhz = {
                firefly_id : firefly_gpu_mhz,
                hikey_id   : hikey_gpu_mhz,
                mate_id    : mate_gpu_mhz,
            }

            # Convolution method mapping
            convolution_method_to_name = [
                'gemm',
                'direct',
                'winograd'
            ]

            # Library.
            library_tags = [ tag for tag in r['dict']['tags'] if tag in tag_to_name.keys() ]
            if len(library_tags)==1:
                library = tag_to_name[library_tags[0]]
            else:
                print('[Warning] Bad library tags. Skipping experiment with tags:')
                print(r['dict']['tags'])
                continue
            if _library and _library!=library: continue

            meta = r['dict']['meta']

            # For each point.
            for point in r['points']:
                point_file_path = os.path.join(r['path'], 'ckp-%s.0001.json' % point)
                with open(point_file_path) as point_file:
                    point_data_raw = json.load(point_file)

                point_env               = point_data_raw['choices']['env']
                characteristics_list    = point_data_raw['characteristics_list']

                num_repetitions = len(characteristics_list)
                platform = model_to_id[point_data_raw['features']['platform']['platform']['model']]
                if _platform and _platform!=platform: continue
                batch_size = np.int64(point_env.get('CK_BATCH_SIZE',-1))
                batch_count = np.int64(point_env.get('CK_BATCH_COUNT',-1))

                convolution_method_from_env = point_env.get('CK_CONVOLUTION_METHOD', point_env.get('CK_CONVOLUTION_METHOD_HINT', 1))
                convolution_method = convolution_method_to_name[np.int64( convolution_method_from_env )]

                data_layout = point_env.get('CK_DATA_LAYOUT','NHWC')
                if library.startswith('tensorflow-') or library.startswith('tflite-'):
                    multiplier = np.float64(point_env.get('CK_ENV_TENSORFLOW_MODEL_MOBILENET_MULTIPLIER',-1))
                    resolution = np.int64(point_env.get('CK_ENV_TENSORFLOW_MODEL_MOBILENET_RESOLUTION',-1))
                    version = np.int64(point_env.get('CK_ENV_TENSORFLOW_MODEL_MOBILENET_VERSION',1))
                else:
                    multiplier_from_env = point_env.get('CK_ENV_MOBILENET_MULTIPLIER', point_env.get('CK_ENV_MOBILENET_WIDTH_MULTIPLIER', -1))
                    multiplier = np.float64( multiplier_from_env )

                    resolution = np.int64(point_env.get('CK_ENV_MOBILENET_RESOLUTION',-1))
                    version = 1
                model = 'v%d-%.2f-%d' % (version, multiplier, resolution)
                cpu_freq = point_data_raw['choices']['cpu_freq']
                gpu_freq = point_data_raw['choices']['gpu_freq']

                dataset_raw = point_env.get('CK_ENV_DATASET_IMAGENET_VAL', '')
                dataset = ''
                if 'val-min-resized' in dataset_raw:
                    dataset = 'val-min-resized'
                elif 'val-min' in dataset_raw:
                    dataset = 'val-min'
                elif 'val' in dataset_raw:
                    dataset = 'val'

                data = []
                for repetition_id, characteristics in enumerate(characteristics_list):
                    datum = {
                        # features
                        'platform': platform,
                        'library': library,
                        # choices
                        'model': model,
                        'batch_size': batch_size,
                        'batch_count': batch_count,
                        'convolution_method': convolution_method,
                        'data_layout': data_layout,
                        'multiplier': multiplier,
                        'resolution': resolution,
                        'version': version,
                        'cpu_freq': cpu_freq,
                        'gpu_freq': gpu_freq,
                        # statistical repetition
                        'repetition_id': repetition_id,
                        # runtime characteristics
                        'success': characteristics['run'].get('run_success', 'n/a'),
                        # meta
                        'os_name': meta['os_name'],
                        'cpu_name': meta['cpu_name'],
                        'gpgpu_name': meta['gpgpu_name'],
                    }
                    if accuracy:
                        datum.update({
                            'accuracy_top1': characteristics['run'].get('accuracy_top1', 0),
                            'accuracy_top5': characteristics['run'].get('accuracy_top5', 0),
                            'dataset': dataset,
                            # 'frame_predictions': characteristics['run'].get('frame_predictions', []),
                        })
                    else:
                        datum.update({
                            'time_avg_ms': characteristics['run']['prediction_time_avg_s']*1e+3,
                            #'time_avg_ms': characteristics['run']['execution_time']*1e+3,
                            #'time_total_ms': characteristics['run']['prediction_time_total_s']*1e+3,
                        })

                    data.append(datum)

                index = [
                    'platform', 'library', 'model', 'multiplier', 'resolution', 'batch_size', 'convolution_method', 'repetition_id'
                ]

                # Construct a DataFrame.
                df = pd.DataFrame(data)
                df = df.set_index(index)
                # Append to the list of similarly constructed DataFrames.
                dfs.append(df)
        if dfs:
            # Concatenate all thus constructed DataFrames (i.e. stack on top of each other).
            result = pd.concat(dfs)
            result.index.names = df.index.names
            result.sort_index(ascending=True, inplace=True)
        else:
            # Construct a dummy DataFrame the success status of which can be safely checked.
            result = pd.DataFrame(columns=['success'])
        return result

    # Return a performance DataFrame with additional accuracy metrics
    def merge_accuracy_to_performance(df_performance, df_accuracy):
        df = df_performance
        accuracy_top1_list, accuracy_top5_list = [], []
        for index, row in df.iterrows():
            (platform, lib, model, multiplier, resolution, batch_size, convolution_method, repetition_id) = index
            row = df_accuracy.loc[(platform, lib, model, multiplier, resolution, batch_size, convolution_method)]
            accuracy_top1_list.append(row['accuracy_top1'][0])
            accuracy_top5_list.append(row['accuracy_top5'][0])
        df = df.assign(accuracy_top1=accuracy_top1_list)
        df = df.assign(accuracy_top5=accuracy_top5_list)
        return df


    # Return a accuracy DataFrame with additional performance metrics
    def merge_performance_to_accuracy(df_performance, df_accuracy):
        df = df_accuracy
        time_avg_min_ms, time_avg_max_ms, time_avg_mean_ms = [], [], []
        time_min_min_ms, time_min_max_ms = [], []
        for index, row in df.iterrows():
            (platform, lib, model, multiplier, resolution, batch_size, convolution_method, repetition_id) = index
            row = df_performance.loc[(platform, lib, model, multiplier, resolution, batch_size, convolution_method)]

            time_avg = row['time_avg_ms']
            time_avg_mean = time_avg.mean()
            time_avg_mean_ms.append(time_avg.mean())
            time_avg_min_ms.append(time_avg.mean() - time_avg.std())
            time_avg_max_ms.append(time_avg.mean() + time_avg.std())
            time_min_min_ms.append(time_avg.min())
            time_min_max_ms.append(time_avg.max())

        df = df.assign(time_avg_min_ms=time_avg_min_ms)
        df = df.assign(time_avg_max_ms=time_avg_max_ms)
        df = df.assign(time_avg_mean_ms=time_avg_mean_ms)
        df = df.assign(time_min_min_ms=time_min_min_ms)
        df = df.assign(time_min_max_ms=time_min_max_ms)
        return df

    def df_as_record(df):
        for index, record in df.to_dict(orient='index').items():
            record.update( {n:v for n,v in zip(df.index.names, index) } )
            yield record

    # prepare table
    selected_repo = i.get('selected_repo', 'mobilenet-v1-armcl-opencl-18.08-52ba29e9') # 'mobilenet-v2-tflite-0.1.7'

    df_acc = get_experimental_results(repo_uoa=selected_repo,
        tags='explore-mobilenets-accuracy', accuracy=True)
    df_perf = get_experimental_results(repo_uoa=selected_repo,
        tags='explore-mobilenets-performance', accuracy=False)


    def to_value(i):
        if type(i) is np.ndarray:
            return i.tolist()

        if isinstance(i, np.int64):
            return int(i)

        if isinstance(i, np.float64):
            return float(i)

        return i

    df_merged = merge_performance_to_accuracy(df_perf, df_acc)

    debug_output = i.get('out')=='con'
    table = []
    for record in df_as_record(df_merged):
        row = {}
        props = [
            'platform',
            'library',
            'model',
            'batch_size',
            'batch_count',
            'convolution_method',
            'data_layout',
            'resolution',
            'multiplier',
            'version',
            'accuracy_top1',
            'accuracy_top5',
            'cpu_freq',
            'gpu_freq',
            'os_name',
            'cpu_name',
            'gpgpu_name',
            'dataset',
        ]
        for prop in props:
            row[prop] = to_value(record.get(prop, ''))

        row['time_avg_ms'] = to_value(record.get('time_avg_mean_ms', ''))
        row['time_avg_ms#min'] = to_value(record.get('time_avg_min_ms', ''))
        row['time_avg_ms#max'] = to_value(record.get('time_avg_max_ms', ''))

        row['time_min_ms'] = to_value(record.get('time_min_min_ms', ''))
        row['time_min_ms#min'] = to_value(record.get('time_min_min_ms', ''))
        row['time_min_ms#max'] = to_value(record.get('time_min_max_ms', ''))

        table.append(row)
        if debug_output:
            ck.out(str(row))

    merged_table = table

    return { 'return': 0, 'table': merged_table }

##############################################################################
# get raw config for repo widget

def get_raw_config(i):
    """
    Input:  {
            }

    Output: {
              return       - return code =  0, if successful
                                         >  0, if error
              (error)      - error text if return > 0
            }

    """

    data_config = cfg['data_config']
    data_config['return'] = 0

    return data_config
