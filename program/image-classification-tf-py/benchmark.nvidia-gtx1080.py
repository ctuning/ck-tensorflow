#! /usr/bin/python
import ck.kernel as ck
import copy
import re
import argparse


# Platform tag.
platform_tags='nvidia-gtx1080'

# Batch size.
bs={
  'start':1,
  'stop':16,
  'step':1,
  'default':1
}

# Number of statistical repetitions e.g. 3.
num_repetitions=3

# Number of batches to run (the first is to be discarded) e.g. 5.
num_batches=5

def do(i, arg):
    # Detect basic platform info.
    ii={'action':'detect',
        'module_uoa':'platform',
        'out':'out'}
    r=ck.access(ii)
    if r['return']>0: return r

    # Host and target OS params.
    hos=r['host_os_uoa']
    hosd=r['host_os_dict']

    tos=r['os_uoa']
    tosd=r['os_dict']
    tdid=r['device_id']

    # Program and command.
    program='image-classification-tf-py'
    cmd_key='default'
    # Load program meta and description to check deps.
    ii={'action':'load',
        'module_uoa':'program',
        'data_uoa':program}
    rx=ck.access(ii)
    if rx['return']>0: return rx
    mm=rx['dict']
    # Get compile-time and run-time deps.
    cdeps=mm.get('compile_deps',{})
    rdeps=mm.get('run_deps',{})
    # Merge rdeps with cdeps for setting up the pipeline (which uses
    # common deps), but tag them as "for_run_time".
    for k in rdeps:
        cdeps[k]=rdeps[k]
        cdeps[k]['for_run_time']='yes'

    # Tensorflow libs.
    depl=copy.deepcopy(cdeps['lib-tensorflow'])
    if (arg.tos is not None) and (arg.did is not None):
        tos=arg.tos
        tdid=arg.did
    ii={'action':'resolve',
        'module_uoa':'env',
        'host_os':hos,
        'target_os':tos,
        'device_id':tdid,
        'out':'con',
        'deps':{'lib-tensorflow':copy.deepcopy(depl)},
        'quiet':'yes'
    }
    r=ck.access(ii)
    if r['return']>0: return r

    udepl=r['deps']['lib-tensorflow'].get('choices',[]) # All UOAs of env for Tensorflow libs.
    if len(udepl)==0:
        return {'return':1, 'error':'no installed Tensorflow libs'}

    # Tensorflow models.
    depm=copy.deepcopy(cdeps['model-and-weights'])
    ii={'action':'resolve',
        'module_uoa':'env',
        'host_os':hos,
        'target_os':tos,
        'device_id':tdid,
        'out':'con',
        'deps':{'tensorflow_model':copy.deepcopy(depm)},
        'quiet':'yes'
    }
    r=ck.access(ii)
    if r['return']>0: return r

    udepm=r['deps']['tensorflow_model'].get('choices',[]) # All UOAs of env for Tensorflow models.
    if len(udepm)==0:
        return {'return':1, 'error':'no installed TensorFlow models'}

    # Load dataset path.
    # FIXME: Does not have to be ImageNet val.
    ii={'action':'show',
        'module_uoa':'env',
        'tags':'dataset,imagenet,val,raw'}
    rx=ck.access(ii)
    if len(rx['lst'])==0: return rx
    # FIXME: Can also be 'CK_ENV_DATASET_IMAGE_DIR'.
    img_dir=rx['lst'][0]['meta']['env']['CK_ENV_DATASET_IMAGENET_VAL']

    # Prepare pipeline.
    cdeps['lib-tensorflow']['uoa']=udepl[0]
    cdeps['model-and-weights']['uoa']=udepm[0]
    ii={'action':'pipeline',
        'prepare':'yes',
        'dependencies':cdeps,

        'module_uoa':'program',
        'data_uoa':program,
        'cmd_key':cmd_key,

        'target_os':tos,
        'device_id':tdid,

        'no_state_check':'yes',
        'no_compiler_description':'yes',
        'skip_calibration':'yes',

        'env':{
          'CK_ENV_DATASET_IMAGE_DIR':img_dir,
          'CK_BATCH_COUNT':num_batches
        },

        'cpu_freq':'max',
        'gpu_freq':'max',

        'flags':'-O3',
        'speed':'no',
        'energy':'no',

        'skip_print_timers':'yes',
        'out':'con'
    }

    r=ck.access(ii)
    if r['return']>0: return r
    fail=r.get('fail','')
    if fail=='yes':
        return {'return':10, 'error':'pipeline failed ('+r.get('fail_reason','')+')'}

    ready=r.get('ready','')
    if ready!='yes':
        return {'return':11, 'error':'pipeline not ready'}

    state=r['state']
    tmp_dir=state['tmp_dir']

    # Remember resolved deps for this benchmarking session.
    xcdeps=r.get('dependencies',{})
    # Clean pipeline.
    if 'ready' in r: del(r['ready'])
    if 'fail' in r: del(r['fail'])
    if 'return' in r: del(r['return'])

    pipeline=copy.deepcopy(r)

    # For each Tensorflow lib.*******************************************************
    for lib_uoa in udepl:
        # Load Tensorflow lib.
        ii={'action':'load',
            'module_uoa':'env',
            'data_uoa':lib_uoa}
        r=ck.access(ii)
        if r['return']>0: return r
        # Get the tags from e.g. 'TensorFlow library (from sources, cuda)'
        lib_name=r['data_name']
        lib_tags=re.match('TensorFlow library \((?P<tags>.*)\)', lib_name)
        lib_tags=lib_tags.group('tags').replace(' ', '').replace(',', '-')
        # Skip some libs with "in [..]" or "not in [..]".
        if lib_tags not in ['prebuilt-cuda', 'fromsources-cuda', 'fromsources-cuda-xla', 'prebuilt-cpu', 'fromsources-cpu', 'fromsources-cpu-xla' ]:
            continue
        cmd_keys = ['default']
        gpu_memory_pc = [50]
        # For each cmd key.*************************************************
        for cmd_key in cmd_keys:
            # For each TensorFlow model.*************************************************
            for model_uoa in udepm:
                # Load Tensorflow model.
                ii={'action':'load',
                    'module_uoa':'env',
                    'data_uoa':model_uoa}
                r=ck.access(ii)
                if r['return']>0: return r
                # Get the tags from e.g. 'TensorFlow python model and weights (squeezenet)'.
                model_name=r['data_name']
                model_tags = re.match('TensorFlow python model and weights \((?P<tags>.*)\)', model_name)
                model_tags = model_tags.group('tags').replace(' ', '').replace(',', '-').lower()
                # Skip some models with "in [..]" or "not in [..]".
                if model_tags not in [ 'squeezenet', 'googlenet', 'alexnet' ]: continue

                record_repo='local'
                record_uoa=model_tags+'-'+lib_tags

                # Prepare pipeline.
                ck.out('---------------------------------------------------------------------------------------')
                ck.out('%s - %s' % (lib_name, lib_uoa))
                ck.out('%s - %s' % (model_name, model_uoa))
                ck.out('Experiment - %s:%s' % (record_repo, record_uoa))

                # Prepare autotuning input.
                cpipeline=copy.deepcopy(pipeline)

                # Reset deps and change UOA.
                new_deps={'lib-tensorflow':copy.deepcopy(depl),
                          'squeezedet':copy.deepcopy(depm)}

                new_deps['lib-tensorflow']['uoa']=lib_uoa
                new_deps['squeezedet']['uoa']=model_uoa

                jj={'action':'resolve',
                    'module_uoa':'env',
                    'host_os':hos,
                    'target_os':tos,
                    'device_id':tdid,
                    'deps':new_deps}
                r=ck.access(jj)
                if r['return']>0: return r

                cpipeline['dependencies'].update(new_deps)

                cpipeline['cmd_key']=cmd_key

                ii={'action':'autotune',

                    'module_uoa':'pipeline',
                    'data_uoa':'program',

                    'choices_order':[
                        [
                            '##choices#env#CK_TF_GPU_MEMORY_PERCENT'
                        ],
                        [
                            '##choices#env#CK_BATCH_SIZE'
                        ]
                    ],
                    'choices_selection':[
                        {'type':'loop', 'choice':gpu_memory_pc},
                        {'type':'loop', 'start':bs['start'], 'stop':bs['stop'], 'step':bs['step'], 'default':bs['default']},
                    ],

                    'features_keys_to_process':['##choices#*'],

                    'iterations':-1,
                    'repetitions':num_repetitions,

                    'record':'yes',
                    'record_failed':'yes',
                    'record_params':{
                        'search_point_by_features':'yes'
                    },
                    'record_repo':record_repo,
                    'record_uoa':record_uoa,

                    'tags':[ 'explore-batch-size-libs-models', model_tags, lib_tags, platform_tags ],

                    'pipeline':cpipeline,
                    'out':'con'}

                r=ck.access(ii)
                if r['return']>0: return r
                fail=r.get('fail','')
                if fail=='yes':
                    return {'return':10, 'error':'pipeline failed ('+r.get('fail_reason','')+')'}

    return {'return':0}

parser = argparse.ArgumentParser(description='Pipeline')
parser.add_argument("--target_os", action="store", dest="tos")
parser.add_argument("--device_id", action="store", dest="did")
myarg=parser.parse_args()


r=do({}, myarg)
if r['return']>0: ck.err(r)
