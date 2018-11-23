#
# Copyright (c) 2018 cTuning foundation.
# See CK COPYRIGHT.txt for copyright details.
#
# SPDX-License-Identifier: BSD-3-Clause.
# See CK LICENSE.txt for licensing details.
#
# Collective Knowledge - raw data access (JSON).
#
# Developers:
# - Nikolay Istomin, Xored.
# - Anton Lokhmotov, dividiti.
# - Leo Gordon, dividiti.
#

cfg={}  # Will be updated by CK (meta description of this module)
work={} # Will be updated by CK (temporal data)
ck=None # Will be updated by CK (initialized CK kernel)

import os
import sys
import json
import re
import traceback
from collections import defaultdict

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

    cpu_code_to_cpu_name_cache = {}

    def map_cpu_code_to_cpu_name( cpu_code ):

        def search_platform_cpu_by( field_name ):
            r = ck.access({ "action":       "search",
                            "module_uoa":   "platform.cpu",
                            "search_dict":  {   "features": {
                                                    field_name: cpu_code,
                                                }
                                            },
                            "add_meta":"yes",
            })
            lst = r.get('lst', [])
            if len(lst)==1:
                return lst[0]
            else:
                return None

        if not cpu_code in cpu_code_to_cpu_name_cache:

            entry = search_platform_cpu_by( 'ck_cpu_name' )
            if not entry:
                entry = search_platform_cpu_by( 'name' )

            if entry:
                cpu_name = entry['meta']['features'].get('ck_arch_real_name', cpu_code) # the entry may be found, but may lack 'ck_arch_real_name'
            else:
                cpu_name = cpu_code     # the entry was not found at all

            cpu_code_to_cpu_name_cache[ cpu_code ] = cpu_name

        return cpu_code_to_cpu_name_cache[ cpu_code ]


    def get_experimental_results(repo_uoa, module_uoa='experiment', tags='explore-mobilenets-accuracy', accuracy=True):

        r = ck.access({'action':'search', 'repo_uoa':repo_uoa, 'module_uoa':module_uoa, 'tags':tags})
        if r['return']>0:
            ck.out('Error: %s' % r['error'])
            exit(1)
        experiments = r['lst']

        tag_to_library_exception = {    # Only add exceptions here!
            'request-d8f69c13':         'armcl-18.03+', # armcl-dv/dt
            '18.05-0acd60ed-request':   'armcl-18.05+',
        }
        platform_config             = cfg['platform_config']
        convolution_method_to_name  = cfg['convolution_method_to_name']

        dfs = []
        for experiment in experiments:
            data_uoa = experiment['data_uoa']
            r = ck.access({'action':'list_points', 'repo_uoa':repo_uoa, 'module_uoa':module_uoa, 'data_uoa':data_uoa})
            if r['return']>0:
                ck.out('Error: %s' % r['error'])
                exit(1)

            library = None
            for tag in r['dict']['tags']:
                if tag in tag_to_library_exception:
                    library = tag_to_library_exception[tag]
                    break
                elif tag.startswith('tflite-') or tag.startswith('tensorflow-'):
                    library = tag
                    break
                else:
                    match_version = re.search('^(\d{2}\.\d{2})-\w+$', tag)
                    if match_version:
                        library = 'armcl-' + match_version.group(1)
                        break

            if not library:
                ck.out('[Warning] Bad library tags: "%s". Skipping...' % str(r['dict']['tags']))
                continue

            meta = r['dict']['meta']

            # For each point.
            for point in r['points']:
                point_file_path = os.path.join(r['path'], 'ckp-%s.0001.json' % point)
                with open(point_file_path) as point_file:
                    point_data_raw = json.load(point_file)

                point_env               = point_data_raw['choices']['env']
                characteristics_list    = point_data_raw['characteristics_list']

                num_repetitions = len(characteristics_list)

                # Platform.
                platform_model = point_data_raw['features']['platform']['platform']['model']
                platform = platform_config.get(platform_model, {'name':platform_model})['name']

                # Batch size and count.
                batch_size = np.int64(point_env.get('CK_BATCH_SIZE',-1))
                batch_count = np.int64(point_env.get('CK_BATCH_COUNT',-1))

                # Convolution method.
                convolution_method_from_env = point_env.get('CK_CONVOLUTION_METHOD', point_env.get('CK_CONVOLUTION_METHOD_HINT',"-1"))
                convolution_method = convolution_method_to_name[ str(convolution_method_from_env) ]

                # Data layout.
                data_layout = point_env.get('CK_DATA_LAYOUT','default')

                # Kernel tuner.
                kernel_tuner = point_env.get('CK_LWS_TUNER_TYPE','default')

                # Model.
                if library.startswith('tensorflow-') or library.startswith('tflite-'):
                    version_from_env    = point_env.get('CK_ENV_TENSORFLOW_MODEL_MOBILENET_VERSION',1)
                    multiplier_from_env = point_env.get('CK_ENV_TENSORFLOW_MODEL_MOBILENET_MULTIPLIER',-1)
                    resolution_from_env = point_env.get('CK_ENV_TENSORFLOW_MODEL_MOBILENET_RESOLUTION',-1)
                else:
                    version_from_env    = point_env.get('CK_ENV_MOBILENET_VERSION',1)
                    multiplier_from_env = point_env.get('CK_ENV_MOBILENET_MULTIPLIER', point_env.get('CK_ENV_MOBILENET_WIDTH_MULTIPLIER', -1))
                    resolution_from_env = point_env.get('CK_ENV_MOBILENET_RESOLUTION',-1)
                version = np.int64(version_from_env)
                multiplier = np.float64(multiplier_from_env)
                resolution = np.int64(resolution_from_env)
                model = 'v%d-%.2f-%d' % (version, multiplier, resolution)

                # Dataset.
                dataset_raw = point_env.get('CK_ENV_DATASET_IMAGENET_VAL', '')
                dataset = ''
                if 'val-min-resized' in dataset_raw:
                    dataset = 'val-min-resized'
                elif 'val-min' in dataset_raw:
                    dataset = 'val-min'
                elif 'val' in dataset_raw:
                    dataset = 'val'

                # Target names for CPU and OS.
                target_os_name = point_data_raw['features']['platform']['os']['name']
                cpu_names = [ map_cpu_code_to_cpu_name(cpu_dict['ck_cpu_name']) for cpu_dict in point_data_raw['features']['platform']['cpu_misc'].values() ]
                cpu_count_by_type = defaultdict(int)
                for cpu_name in cpu_names:
                    cpu_count_by_type[cpu_name] += 1
                target_cpu_name = ' + '.join( [ '{} MP{}'.format(k,v) for (k,v) in cpu_count_by_type.items() ] )

                # Frequencies for CPU and GPU.
                cpu_freq = point_data_raw['choices']['cpu_freq']
                gpu_freq = point_data_raw['choices']['gpu_freq']

                # Construct an individual DataFrame.
                data = []
                for repetition_id, characteristics in enumerate(characteristics_list):
                    datum = {
                        # features
                        'platform': platform,
                        'library': library,
                        # choices
                        'model': model,
                        'version': version,
                        'multiplier': multiplier,
                        'resolution': resolution,
                        'batch_size': batch_size,
                        'batch_count': batch_count,
                        'convolution_method': convolution_method, # 3 choices: DEFAULT, GEMM, DIRECT
                        'data_layout': data_layout,               # 2 choices: NCHW, NHWC
                        'kernel_tuner': kernel_tuner,             # 2 choices: NONE, DEFAULT
                        'tunerXlayout':  '%s X %s' % (kernel_tuner, data_layout),        # 2X2 choices
                        'methodXlayout': '%s X %s' % (convolution_method, data_layout),  # 3X2 choices
                        'methodXtuner':  '%s X %s' % (convolution_method, kernel_tuner), # 3X2 choices
                        'cpu_freq': cpu_freq,
                        'gpu_freq': gpu_freq,
                        # statistical repetition
                        'repetition_id': repetition_id,
                        # runtime characteristics
                        'success': characteristics['run'].get('run_success', 'n/a'),
                        # meta
                        'os_name': target_os_name,
                        'cpu_name': target_cpu_name,
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
                            'time_avg_ms': 1e+3*characteristics['run'].get('prediction_time_avg_s',0.0)
                            #'time_total_ms': characteristics['run']['prediction_time_total_s']*1e+3,
                        })

                    data.append(datum)

                index = [
                    'platform', 'library',
                    'model', 'version', 'multiplier', 'resolution',       # model
                    'batch_size',                                         # TODO: batch_count?
                    'convolution_method', 'data_layout', 'kernel_tuner',  # ArmCL specific
                    'tunerXlayout', 'methodXlayout', 'methodXtuner',      # combined keys
                    'repetition_id'
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


    # Return an accuracy DataFrame with additional performance metrics.
    def merge_performance_to_accuracy(df_performance, df_accuracy):
        df = df_accuracy
        # Show variation appropriate for each execution time metric:
        # mean-std..mean+std for time_avg_ms, min..max for time_min_ms.
        time_avg_min_ms, time_avg_max_ms, time_avg_mean_ms = [], [], []
        time_min_min_ms, time_min_max_ms = [], []
        # Iterate over the indices of the accuracy DataFrame and
        # find corresponding rows in the performance DataFrame.
        for index, _ in df.iterrows():
            # Catch abnormal situation when no corresponding performance data is available.
            try:
                # Chop off the last key (repetition_id).
                row = df_performance.loc[index[:-1]]
            except:
                row = None
                ck.out('[Warning] Found no performance data corresponding to accuracy data with index: "%s". Plotting at zero time...' % str(index))
#                # Show trace.
#                ck.out('-'*80)
#                traceback.print_exc()
#                ck.out('-'*80)
#                # Show sample index structure for performance data.
#                for index_perf, _ in df_performance.iterrows():
#                    ck.out('- performance data index: %s' % str(index_perf))
#                    break

            if row is not None:
                time_avg = row['time_avg_ms']
                time_avg_mean_ms.append(time_avg.mean())
                # NB: Setting ddof=0 avoids getting nan when there's only one repetition.
                time_avg_min_ms.append(time_avg.mean() - time_avg.std(ddof=0))
                time_avg_max_ms.append(time_avg.mean() + time_avg.std(ddof=0))
                time_min_min_ms.append(time_avg.min())
                time_min_max_ms.append(time_avg.max())
            else:
                time_avg_mean_ms.append(0)
                time_avg_min_ms.append(0)
                time_avg_max_ms.append(0)
                time_min_min_ms.append(0)
                time_min_max_ms.append(0)

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

    default_selected_repo = ''
    default_selected_repo = 'mlperf-mobilenets'
    default_selected_repo = 'linaro-hikey960-18.08-52ba29e9-mobilenet-v1-0.25-128'

    selected_repo = i.get('selected_repo', default_selected_repo)

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
            'dataset',
            'model',
            'version',
            'multiplier',
            'resolution',
            'batch_size',
            'batch_count',
            'convolution_method',
            'data_layout',
            'kernel_tuner',
            'tunerXlayout',
            'methodXlayout',
            'methodXtuner',
            'accuracy_top1',
            'accuracy_top5',
            'os_name',
            'cpu_name',
            'gpgpu_name',
            'cpu_freq',
            'gpu_freq',
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
