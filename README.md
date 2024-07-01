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
