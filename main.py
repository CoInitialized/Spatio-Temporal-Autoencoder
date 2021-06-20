import torch
import torch.nn as nn
import torch.optim as optim
from net import SpatialEncoder
from utils import *
import tqdm
import gc
from pathlib import Path
import numpy as np
import argparse
import pickle
 
 
parser = argparse.ArgumentParser()
parser.add_argument('--threshold', type=float, default=0.7, required= False)
parser.add_argument('--path', type=str)
parser.add_argument('--train', type=bool, required=False)
parser.add_argument('--output_path', type=str, required=False)
parser.add_argument('--model_path', type=str, required=False)
 
 
 
 
def calculate_score(input, output):
 
    abnormality_scores = torch.sqrt(torch.sum(torch.square(input - output), axis = (-1,-2)))
    minimum, maximum = min(abnormality_scores), max(abnormality_scores)
    abnormality_scores = (abnormality_scores - minimum) / maximum       
    return 1 - abnormality_scores
 
 
 
def main():
 
    args = parser.parse_args()
 
    if args.train == True:
        pass
    else:
        test_path = Path(args.path)
        model_path = Path(args.model_path)
        threshold = args.threshold
        out_path = args.output_path
 
 
 
        model = SpatialEncoder()
        optimizer = optim.Adam(model.parameters())
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
 
 
 
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
 
        model = model.to(device)
        model = model.eval()
 
        files = glob.glob(str(test_path) + '/*')
        read_files = [load_data_from_file(file) for file in files]
        data_dict = {name : generate_stride_set(video_array, 10,10) for name, video_array in zip(files, read_files)}
        results_dict = {}
        out_of_threshold_dict = {}
        for name, stride_set in data_dict.items():
                print(name)
                predictions = []
                for window in stride_set:
                    
                    window = window.to(device)
                    window = torch.unsqueeze(torch.unsqueeze(window, 0),0)
                    out = model(window)
                    out = out.cpu().detach().numpy()
                    predictions.append(out)
 
                predictions = torch.cat([torch.Tensor(x) for x in list(np.asarray(predictions).squeeze())]).reshape(-1,227,227)
                inputs = data_dict[name].reshape(-1,227,227)
                scores = calculate_score(inputs, predictions)
                abnormal = np.where(np.array(scores) < threshold)
                results_dict[name] = scores
                out_of_threshold_dict[name] = abnormal
        
        if out_path:
            with open(out_path + '/data_dict.pkl', 'wb') as data_dict_file, open(out_path + '/results_dict.pkl', 'wb') as results_dict_file, open(out_path + '/out_of_threshold_dict.pkl', 'wb') as out_of_threshold_file: 
                pickle.dump(data_dict, data_dict_file)
                pickle.dump(results_dict, results_dict_file)
                pickle.dump(out_of_threshold_dict, out_of_threshold_file)
 
 
if __name__ == '__main__':
    main()
