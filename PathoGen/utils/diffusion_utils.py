import numpy as np
import torch
import os
import sys
from torch._C import _get_float32_matmul_precision
from tqdm import tqdm
from argparse import Namespace
from torchmetrics.regression import SpearmanCorrCoef, MeanAbsoluteError
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import pandas as pd

mms_list = joblib.load('Data/TCGA_KIRC/org_minmax_scaler.pkl')

def zscore(tensor):
    mean = torch.mean(tensor, dim=0, keepdim=True)
    std = torch.std(tensor, dim=0, keepdim=True)
    z_score_normalized_tensor = (tensor - mean) / std
    return z_score_normalized_tensor


def train_loop(args: Namespace, cur, epoch, model, loader, optimizer, writer):   
    
    train_loss = 0.
    print('\n')
    
    for batch_idx, (slide_id, level, data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6) in tqdm(enumerate(loader)):

        data_WSI = data_WSI.to(model.device)
        
        data_omic1 = data_omic1.type(torch.FloatTensor).to(model.device)[0]
        data_omic2 = data_omic2.type(torch.FloatTensor).to(model.device)[0]
        data_omic3 = data_omic3.type(torch.FloatTensor).to(model.device)[0]
        data_omic4 = data_omic4.type(torch.FloatTensor).to(model.device)[0]
        data_omic5 = data_omic5.type(torch.FloatTensor).to(model.device)[0]
        data_omic6 = data_omic6.type(torch.FloatTensor).to(model.device)[0]

        t = torch.randint(0, args.T, (args.batch_size,), device=model.device).long()

        loss  = model.p_losses([data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6], data_WSI, t, args.lambda_regen, args.gc)
        
        loss.backward()

        if (batch_idx + 1) % args.gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

        train_loss += loss.item()*args.gc

    # calculate loss and error for epoch
    train_loss /= len(loader)

    if epoch % args.save_epoch == 0 or epoch == args.max_epochs-1:
        torch.save(model.model.state_dict(), args.results_dir+'/'+"w"+str(epoch)+'.pt')

    #save and print result
    message = f"{epoch}/{args.max_epochs} -- Train Loss: {train_loss}"
    print(message)
    with open(writer, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message


def validate_loop(args, cur, epoch, model, loader, writer):
    
    model.model.eval()
    scc = SpearmanCorrCoef().to(model.device)
    val_scores = [0.0] * 7

    for batch_idx, data in tqdm(enumerate(loader)):
        data = [d.to(model.device) if i == 0 else d.type(torch.FloatTensor).to(model.device) for i, d in enumerate(data)]
        data_WSI, *data_omics = data

        with torch.no_grad():
            gen_omic = model.sample(data_WSI, model.model.omic_sizes, args.batch_size)

        gen_omic = [zscore(g) for g in gen_omic]

        scc_scores = [scc(gen_omic[i], data_omics[i][0]) for i in range(6)]
        for i in range(6):
            val_scores[i] += float(scc_scores[i])
        val_scores[6] += float(sum(scc_scores) / 6)

    val_scores = [score / len(loader) for score in val_scores]

    message = f"{epoch}/{args.max_epochs} -- Validation -- " + " -- ".join([f"G{i+1}: {val_scores[i]}" for i in range(6)]) + f" -- G: {val_scores[6]}"
    print(message)
    with open(writer, "a") as log_file:
        log_file.write('%s\n' % message)

    return val_scores[6]

def compute_score(pred_omic, gt_omic, scc, mae):
    pdo = torch.cat(pred_omic, dim=-1)
    gto = torch.cat(gt_omic, dim=-1)
    scc_scores = [float(scc(pred_omic[i], gt_omic[i])) for i in range(6)]
    mae_scores = [float(mae(pred_omic[i], gt_omic[i])) for i in range(6)]
    scc_scores.append(float(scc(pdo, gto)))
    mae_scores.append(float(mae(pdo, gto)))
    return scc_scores, mae_scores

def merge_tensors(pred):

    num_tensors = len(pred[0])

    # Initialize a list to hold the concatenated tensors
    concatenated_tensors = []
    for i in range(num_tensors):
        # Concatenate tensors at position i from all sublists
        concatenated_tensor = torch.cat([sublist[i] for sublist in pred])
        concatenated_tensors.append(concatenated_tensor)  

    return  concatenated_tensors 

def check_equal(gen_omic1, gen_omic2, gen_omic3):
    if len(gen_omic1) != len(gen_omic2) or len(gen_omic1) != len(gen_omic3):
        return False

    for t1, t2, t3 in zip(gen_omic1, gen_omic2, gen_omic3):
        if not torch.equal(t1, t2) or not torch.equal(t1, t3):
            return False

    return True

def compute_mean_tensors(list_of_lists):
    num_lists = len(list_of_lists)
    num_tensors = len(list_of_lists[0])
    
    # Initialize a list to store the sums of the tensors
    sum_tensors = [torch.zeros_like(list_of_lists[0][i]) for i in range(num_tensors)]
    
    # Sum the tensors
    for lst in list_of_lists:
        for i in range(num_tensors):
            sum_tensors[i] += lst[i]
    
    # Compute the mean tensors
    mean_tensors = [sum_tensors[i] / num_lists for i in range(num_tensors)]
    
    return mean_tensors




def test_loop(args, epoch, model, loader, writer):
    
    wt = torch.load(args.weight_path, map_location=torch.device('cpu'), weights_only=False)
    model.model.load_state_dict(wt)
    
    model.model.eval()

    pred1, pred2, pred3, gt1, gt2, pred4, gt3, sid = [], [], [], [], [], [], [], []
    scc = SpearmanCorrCoef().to(model.device)
    mae = MeanAbsoluteError().to(model.device)

    for batch_idx, data in tqdm(enumerate(loader)):
        data = [d if i in [0,1,2] else d.type(torch.FloatTensor).to(model.device)[0] for i, d in enumerate(data)]
        slide_id, level, data_WSI, *data_omics = data
        data_WSI = data_WSI.to(model.device)

        with torch.no_grad():
            gen_omic1 = model.sample_clamp(data_WSI, model.model.omic_sizes, args.batch_size)
            gen_omic2 = model.sample_clamp(data_WSI, model.model.omic_sizes, args.batch_size)
            gen_omic3 = model.sample_clamp(data_WSI, model.model.omic_sizes, args.batch_size)

            gen_omic = [(g1+g2+g3)/3 for g1, g2, g3 in zip(gen_omic1, gen_omic2, gen_omic3)]

        gt1.append(data_omics)
        
        gen_omic = [torch.tensor(mms_list[i].transform([zscore(g).cpu()]))[0].to(model.device) for i, g in enumerate(gen_omic)]
        pred2.append(gen_omic)
        sid.append(slide_id)

        gen_omic = [torch.tensor(mms_list[i].inverse_transform([g.cpu()]))[0].to(model.device) for i, g in enumerate(gen_omic)]
        data_omic = [torch.tensor(mms_list[i].inverse_transform([g.cpu()]))[0].to(model.device) for i, g in enumerate(data_omics)]
        pred3.append(gen_omic)
        gt2.append(data_omic)
        

    pred_dict = dict(zip(sid, pred2))
    torch.save(pred_dict, args.results_dir+'epoch'+str(epoch)+'pred_cal.pt')

    cor_mean2, mae_mean2 = compute_score(merge_tensors(pred2), merge_tensors(gt1), scc, mae)
    cor_mean3, mae_mean3 = compute_score(merge_tensors(pred3), merge_tensors(gt2), scc, mae)

    gene_type = ['Tumor Suppressor Genes', 'Oncogenes', 'Protein Kinases', 'Cell Differentiation Markers', 'Transcription Factors', 'Cytokines and Growth Factors', 'All Genes']

    df = pd.DataFrame({
        'Gene Type': gene_type,
        'MAE': mae_mean2,
        'Correlation': cor_mean2
    })

    file_path = '/content/drive/MyDrive/Histopathology/Code/PG/results/1000T_8gc_mms/P2G_results_'+str(epoch)+'_cal.xlsx'
    df.to_excel(file_path, index=False)


    message = f"\nOverall result -> \nnorm ->  \n spearman: {cor_mean2} \n MAE: {mae_mean2}"
    message += f"original -> \n spearman: {cor_mean3} \n MAE: {mae_mean2} \n"
    print(message)
    with open(writer, "a") as log_file:
        log_file.write('%s\n' % message)


def save_prediction(args, epoch, model, loader):
    
    wt = torch.load(args.weight_path, map_location=torch.device('cpu'), weights_only=False)
    model.model.load_state_dict(wt)
    
    model.model.eval()

    prediction = []

    for batch_idx, data in tqdm(enumerate(loader)):
        data = [d if i in [0,1,2] else d.type(torch.FloatTensor).to(model.device)[0] for i, d in enumerate(data)]
        slide_id, level, data_WSI, *data_omics = data
        data_WSI = data_WSI.to(model.device)

        with torch.no_grad():
            gen_omic1 = model.sample_clamp(data_WSI, model.model.omic_sizes, args.batch_size)
            gen_omic2 = model.sample_clamp(data_WSI, model.model.omic_sizes, args.batch_size)
            gen_omic3 = model.sample_clamp(data_WSI, model.model.omic_sizes, args.batch_size)

            gen_omic = [(g1+g2+g3)/3 for g1, g2, g3 in zip(gen_omic1, gen_omic2, gen_omic3)]

        prediction.append({
            'slide_id': slide_id,
            'gt_omic': data_omics,
            'pred_omic': gen_omic
        })

    torch.save(prediction, args.results_dir+'epoch'+str(epoch)+'test_pred_sm.pt')
    print('\nSaved.')


