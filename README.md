# Spatiotemporal-Autoencoder-Anomaly-Detection

## Requiered packages:
 - torch
 - tqdm
 - gc
 - pathlib 
 - numpy
 - argparse
 - pickle
 - cassandra
 - diskcache
 - io
 - scipy
 - pandas
 - matplotlib
 - PIL
 - wandb
## Run experiments

For running experiments we reccomend usage of main.py file. It will use model trained by us, to predict anomalies in test videos. <br>
Example of main.py usage: <br>

```cmd
python main.py --threshold 0.7\
--path "<PATH>\SpatiotemporalAutoencoderAnomalyDetection\Avenue Dataset\testing_vol"\  
--output_path "<PATH>\SpatiotemporalAutoencoderAnomalyDetection\output"\  
--model_path "<PATH>\SpatiotemporalAutoencoderAnomalyDetection\checkpoints\1_model.pt"
```
This will create on output file, which is python pickle. It can be loaded for further analysis. It should countain predicted parts of the videos that are anomalous. 
