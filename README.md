# selfies

## Common cmds
```
!conda env export | grep -v "^prefix: " > environment.yml 
```


## Code snippets

### AIS
``` #  gra 2024-07-01
git clone https://github.com/snu-lcbc/atom-in-SMILES
cd atom-in-SMILES
python setup.py install

pip install deepchem
```
 
### CPU

#### Inspect the specs of each core
```
!cat /proc/cpuinfo 
```

#### RAM 
```
!cat /proc/meminfo | grep "MemTotal"
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


#### Fast Parquet 

```
from fastparquet import write
write('outfile.parq', df)
write('outfile2.parq', df, row_group_offsets=[0, 10000, 20000],
      compression='GZIP', file_scheme='hive')
```
#### GPU specification

- https://docs.alliancecan.ca/wiki/Using_GPUs_with_Slurm/en

</details>



### Kaggle

| Hardware Component | Release Year | Core Count | Memory | Hours per Week |
| :-- | :-- | :-- | :-- | :-- |
| Intel Xeon CPU 2.00 GHz CPU | 2012 | 4 vCPU cores | 32 GB | Unlimited |
| NVIDIA Tesla P100 GPU | 2016 | 3584 Cuda cores| 16 GB | 30 h |
| Google TPU v3-8 | 2018 | 8 TPU v3 cores | 128 GB | 20 h, only 8, so need to wait in the queue |

### LGB with GPU enabled
``` # did not work on gra 2024-07-01
git clone --recursive https://github.com/microsoft/LightGBM.git
cd LightGBM
sh ./build-python.sh install --gpu
```




### Majority voting
```
class VotingModel(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators        
    def fit(self, X, y=None):
        return self    
    def predict(self, X):
        y_preds = [estimator.predict(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)    
    def predict_proba(self, X):
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)    
```



## Padding - efficient

```
pad_end = lambda a,i=600: a[0:i] if len(a) > i else a + [0] * (i-len(a))
pad_start = lambda a,i=400: a[0:i] if len(a) > i else [0] * (i-len(a))+a
```

```
def center_pad( a, N=500 ):   
    m = len(a)    
    n = np.abs(N - m)//2 
    p = [0]*n+a+[0]*n if m<N else a[n:-n]
    # uncomment below if all must have exact length N
    #if (N-len(p))>0:
    #    p=[0]+p
    return p  
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


<details>
<summary>R</summary>
  
```
install.packages("devtools")
devtools::install_github("fangzhou-xie/rethnicity")
```
  
</details>
