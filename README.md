

# A Circular Window-based Cascade Transformer for Online Action Detection

PyTorch implementation of "A Circular Window-based Cascade Transformer for Online Action Detection".

## Data Preparation

1. Download datasets: [`THUMOS'14`](https://www.crcv.ucf.edu/THUMOS14/) and [`TVSeries`](https://homes.esat.kuleuven.be/psi-archive/rdegeest/TVSeries.html)
2. Extract video frames at 24 FPS
3. Construct target files and feature files following [LSTR](https://github.com/amazon-science/long-short-term-transformer) method
4. Organize data structure:

    ```
    $DATA_ROOT
    ├── rgb_kinetics_resnet50/          # RGB features (L x 2048)
    ├── flow_kinetics_bninception/       # Flow features (L x 1024)  
    └── target_perframe/                 # Labels (L x 22)
    ```

## Training

Train the model using distributed training across multiple GPUs:

```shell
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
PATH_TO_CONFIG_FILE=""
PORT=$(shuf -i 25000-35000 -n 1)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --nproc_per_node=4 --master_port=$PORT tools/train_net.py \
  --config_file $PATH_TO_CONFIG_FILE
```

## Evaluation

Evaluate the trained model on the test dataset:

```shell
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
PATH_TO_CONFIG_FILE=""
CUDA_VISIBLE_DEVICES=0 python tools/test_net.py \
  --config_file $PATH_TO_CONFIG_FILE
```

## Acknowledgments

Built upon [LSTR](https://github.com/amazon-science/long-short-term-transformer).
