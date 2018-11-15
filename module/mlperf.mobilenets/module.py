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
            }

    Output: {
              return       - return code =  0, if successful
                                         >  0, if error
              (error)      - error text if return > 0
            }

    """

    repo_uoa = 'ck-request-asplos18-mobilenets-armcl-opencl-accuracy-500'

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
                # ArmCL tags on Firefly.
                '17.12-48bc34e'     : 'armcl-17.12',
                '18.01-f45d5a9'     : 'armcl-18.01',
                '18.03-e40997b'     : 'armcl-18.03',
                '18.05-b3a371b'     : 'armcl-18.05',
                # TensorFlow tags.
                'tensorflow-1.7'    : 'tensorflow-1.7',
                'tensorflow-1.8'    : 'tensorflow-1.8',
            }

            # Platforms
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

            # Platform mappings
            model_to_id = {
                firefly_model : firefly_id,
                hikey_model   : hikey_id
            }
            id_to_name = {
                firefly_id : firefly_name,
                hikey_id   : hikey_name
            }
            id_to_gpu = {
                firefly_id : firefly_gpu,
                hikey_id   : hikey_gpu
            }
            id_to_gpu_mhz = {
                firefly_id : firefly_gpu_mhz,
                hikey_id   : hikey_gpu_mhz
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
            # For each point.
            for point in r['points']:
                point_file_path = os.path.join(r['path'], 'ckp-%s.0001.json' % point)
                with open(point_file_path) as point_file:
                    point_data_raw = json.load(point_file)
                characteristics_list = point_data_raw['characteristics_list']
                num_repetitions = len(characteristics_list)
                platform = model_to_id[point_data_raw['features']['platform']['platform']['model']]
                if _platform and _platform!=platform: continue
                batch_size = np.int64(point_data_raw['choices']['env'].get('CK_BATCH_SIZE',-1))
                batch_count = np.int64(point_data_raw['choices']['env'].get('CK_BATCH_COUNT',-1))
                convolution_method = convolution_method_to_name[np.int64(point_data_raw['choices']['env'].get('CK_CONVOLUTION_METHOD_HINT',1))]
                if library.startswith('tensorflow-'):
                    multiplier = np.float64(point_data_raw['choices']['env'].get('CK_ENV_TENSORFLOW_MODEL_MOBILENET_MULTIPLIER',-1))
                    resolution = np.int64(point_data_raw['choices']['env'].get('CK_ENV_TENSORFLOW_MODEL_MOBILENET_RESOLUTION',-1))
                else:
                    multiplier = np.float64(point_data_raw['choices']['env'].get('CK_ENV_MOBILENET_WIDTH_MULTIPLIER',-1))
                    resolution = np.int64(point_data_raw['choices']['env'].get('CK_ENV_MOBILENET_RESOLUTION',-1))
                model = 'v1-%.2f-%d' % (multiplier, resolution)
                if accuracy:
                    data = [
                        {
                            # features
                            'platform': platform,
                            'library': library,
                            # choices
                            'model': model,
                            'batch_size': batch_size,
                            'batch_count': batch_count,
                            'convolution_method': convolution_method,
                            'resolution': resolution,
                            'multiplier': multiplier,
                            # statistical repetition
                            'repetition_id': repetition_id,
                            # runtime characteristics
                            'success': characteristics['run'].get('run_success', 'n/a'),
                            'accuracy_top1': characteristics['run'].get('accuracy_top1', 0),
                            'accuracy_top5': characteristics['run'].get('accuracy_top5', 0),
                            'frame_predictions': characteristics['run'].get('frame_predictions', []),
                        }
                        for (repetition_id, characteristics) in zip(range(num_repetitions), characteristics_list)
                    ]
                else: # performance
                    data = [
                        {
                            # features
                            'platform': platform,
                            'library': library,
                            # choices
                            'model': model,
                            'batch_size': batch_size,
                            'batch_count': batch_count,
                            'convolution_method': convolution_method,
                            'resolution': resolution,
                            'multiplier': multiplier,
                            # statistical repetition
                            'repetition_id': repetition_id,
                            # runtime characteristics
                            'success': characteristics['run'].get('run_success', 'n/a'),
                            'time_avg_ms': characteristics['run']['prediction_time_avg_s']*1e+3,
                            'time_total_ms': characteristics['run']['prediction_time_total_s']*1e+3,
                        }
                        for (repetition_id, characteristics) in zip(range(num_repetitions), characteristics_list)
                    ]
                index = [
                    'platform', 'library', 'model', 'multiplier', 'resolution', 'batch_size', 'convolution_method', 'repetition_id'
                ]

                # HACK: Backup data with another key because set_index makes this data unavailable
                for datum in data:
                    for key in index:
                        datum['_' + key] = datum[key]

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

    # Return a new DataFrame with only the performance and accuracy metrics.
    def merge_performance_accuracy(df_performance, df_accuracy, 
                                   reference_platform=None, reference_lib=None, reference_convolution_method='direct',
                                   performance_metric='time_avg_ms', accuracy_metric='accuracy_top1'):
        df = df_performance[[performance_metric]]
        accuracy_list = []
        for index, row in df.iterrows():
            (platform, lib, model, multiplier, resolution, batch_size, convolution_method) = index
            if reference_platform: platform = reference_platform
            try:
                accuracy = df_accuracy.loc[(platform, lib, model, multiplier, resolution, batch_size, convolution_method)][accuracy_metric]
            except:
                if reference_lib: lib = reference_lib
                convolution_method = reference_convolution_method
                accuracy = df_accuracy.loc[(platform, lib, model, multiplier, resolution, batch_size, convolution_method)][accuracy_metric]
            accuracy_list.append(accuracy)
        df = df.assign(accuracy_top1=accuracy_list) # FIXME: assign to the value of accuracy_metric
        return df

    # prepare table
    df = get_experimental_results(repo_uoa=repo_uoa)

    def to_value(i):
        if type(i) is np.ndarray:
            return i.tolist()

        if isinstance(i, np.int64):
            return int(i)

        if isinstance(i, np.float64):
            return float(i)

        return i

    props = [
        '_platform',
        '_library',
        '_model',
        '_batch_size',
        'batch_count',
        '_convolution_method',
        '_resolution',
        '_multiplier',
        '_repetition_id',
        'success',
        'time_avg_ms',
        'time_total_ms',
    ]

    table = []
    # for record in df.groupby(level=df.index.names[:-1]):
    # for record in df.to_dict(orient='records'):
    for record in df.to_dict(orient='records'):
        row = {}
        for prop in props:
            row[prop] = to_value(record.get(prop, ''))

        table.append(row)

    #     # energies = [ iteration['energy'] for iteration in record['report']['iterations'] ]
    #     # fevs = list(range(len(energies)))
    #     # last_energy = energies[-1]
    #     # minimizer_method = record.get('_minimizer_method', '')
    #     # last_fev = row['nfev']-1 if minimizer_method=='my_cobyla' or 'my_nelder_mead' else row['nfev']

    #     # row['__energies'] = energies
    #     # row['__fevs'] = fevs

    #     # row['##data_uid'] = "{}:{}".format(record['_point'], record['_repetition_id'])

    #     table.append(row)

    return { 'return': 0, 'table': table }

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
