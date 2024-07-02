# selfies

## Common cmds
```
!conda env export | grep -v "^prefix: " > environment.yml 
```


## Code snippest

### AIS
``` #  gra 2024-07-01
git clone https://github.com/snu-lcbc/atom-in-SMILES
cd atom-in-SMILES
python setup.py install

pip install deepchem
```

### GPU

<details>

#### tf

```
import tensorflow as tf

try:
  tpus = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu="local")
except:
  gpus = tf.config.experimental.list_physical_devices('GPU')  
```

#### GPU specification

- https://docs.alliancecan.ca/wiki/Using_GPUs_with_Slurm/en

</details>

### LGB with GPU enabled
``` # did not work on gra 2024-07-01
git clone --recursive https://github.com/microsoft/LightGBM.git
cd LightGBM
sh ./build-python.sh install --gpu
```




## Parquet

```
from pyarrow.parquet import ParquetFile
import pyarrow as pa 

pf = ParquetFile(train_path) 
first_ten_rows = next(pf.iter_batches(batch_size = 10)) 
df = pa.Table.from_batches([first_ten_rows]).to_pandas() 
df.head()
```

