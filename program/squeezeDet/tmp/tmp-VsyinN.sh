#! /bin/bash


export PATH=/home/fanran/CK/ck-env/platform.init/generic-linux-dummy:$PATH


. /home/fanran/CK/local/env/a7829380ccd3a7b4/env.sh
. /home/fanran/CK/local/env/22b9f838712c01ec/env.sh
. /home/fanran/CK/local/env/4ece8b3db3d3aa2d/env.sh
. /home/fanran/CK/local/env/0ff4fb0a2cfd9086/env.sh
. /home/fanran/CK/local/env/41ea79b2343a28ec/env.sh



export CK_DATASET_PATH=/home/fanran/CK/ck-tensorflow/dataset/squeezedet-demo/

export CK_DATASET_FILENAME=squeezedet_demo.sh

echo    executing code ...
 sh /home/fanran/CK/ck-tensorflow/dataset/squeezedet-demo/squeezedet_demo.sh 
