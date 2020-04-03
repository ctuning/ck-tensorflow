## Install dependencies on Ubuntu

https://www.tensorflow.org/install/install_sources

### Install Python 3 (recommended) or Python 2 
```bash
$ sudo apt-get install libcupti-dev
$ sudo apt-get install python-dev python-pip python-wheel
```

### Install Python packages (in user-space)
```bash
python -m pip install --user absl-py==0.8.0              # >=0.7.0
python -m pip install --user astor==0.7.1                # >=0.6.0
python -m pip install --user cachetools==4.0.0           # <5.0,>=2.0.0
python -m pip install --user gast==0.2.2                 # exact
python -m pip install --user google-auth==1.13.1         # <2,>=1.6.3
python -m pip install --user google-auth-oauthlib==0.4.1 # <0.5,>=0.4.1
python -m pip install --user google-pasta==0.2.0         # >=0.1.6
python -m pip install --user grpcio==1.27.2              # >=1.8.6
python -m pip install --user h5py==2.10.0                # exact?
python -m pip install --user keras-applications==1.0.8   # >=1.0.8
python -m pip install --user keras-preprocessing==1.1.0  # >=1.0.5
python -m pip install --user markdown==3.2.1             # >=2.6.8
python -m pip install --user numpy==1.17.2               # <2.0,>=1.16.0
python -m pip install --user oauthlib==3.1.0             # ?
python -m pip install --user opt-einsum==3.2.0           # >=2.3.2
python -m pip install --user protobuf==3.6.1             # >=3.6.1
python -m pip install --user pyasn1==0.4.8               # <0.5.0,>=0.4.6
python -m pip install --user pyasn1-modules==0.2.8       # >=0.2.1
python -m pip install --user requests==2.23.0            # <3,>=2.21.0
python -m pip install --user requests-oauthlib==1.3.0    # >=0.7.0
python -m pip install --user rsa==4.0                    # <4.1,>=3.1.4
python -m pip install --user setuptools==46.1.3          # >=41.0.0
python -m pip install --user tensorboard==2.0.2          # <2.1.0,>=2.0.0 (almost exact)
python -m pip install --user tensorflow-estimator==2.0.1 # <2.1.0,>=2.0.0 (almost exact)
python -m pip install --user termcolor==1.1.0            # >=1.1.0
python -m pip install --user wheel==0.30.0               # >=0.26
python -m pip install --user werkzeug==1.0.1             # >=0.11.15
python -m pip install --user wrapt==1.11.2               # >=1.11.1
```

## Prevent running out of memory

To prevent running out of memory during a build, restrict the build to use
e.g. 1 processor:
```bash
$ ck install package:lib-tensorflow-2.0.1-src-cuda --env.CK_HOST_CPU_NUMBER_OF_PROCESSORS=1
```

## Known problems
### CUDA 10.2 in not supported

[Downgrade to 10.1](https://github.com/tensorflow/tensorflow/issues/34429).
