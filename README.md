# selfies

## Common cmds
```
!conda env export | grep -v "^prefix: " > environment.yml 
```


## Code snippest

```
import tensorflow as tf

try:
  tpus = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu="local")
except:
  gpus = tf.config.experimental.list_physical_devices('GPU')  
```


## Building LGB with GPU enabled
``` # did not work on gra 2024-07-01
git clone --recursive https://github.com/microsoft/LightGBM.git
cd LightGBM
sh ./build-python.sh install --gpu
```


## Building AIS
``` #  gra 2024-07-01
git clone https://github.com/snu-lcbc/atom-in-SMILES
cd atom-in-SMILES
python setup.py install
```
