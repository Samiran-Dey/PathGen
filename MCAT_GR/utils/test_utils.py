import numpy as np
import torch
import pickle 
from utils.utils import *
import os
from collections import OrderedDict
from tqdm import tqdm
import pandas as pd
from collections import Counter
from argparse import Namespace
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import cohen_kappa_score, roc_auc_score, balanced_accuracy_score, f1_score, roc_curve, auc

def test_surv_grad(args, model, loader, writer=None, results_dir=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    pred_grad, true_grad, pred_prob, true_prob, result = [], [], [], [], []
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))
    attention={}

    for batch_idx, (slide_id, level, data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label_g, label_s, event_time, c) in tqdm(enumerate(loader)):
        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        label_g = label_g.type(torch.FloatTensor).to(device)
        label_s = label_s.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)

        with torch.no_grad():
            

            grad, surv, hazards, S, Y_hat, A = model(
                x_path=data_WSI, 
                x_omic1=data_omic1, 
                x_omic2=data_omic2, 
                x_omic3=data_omic3, 
                x_omic4=data_omic4, 
                x_omic5=data_omic5, 
                x_omic6=data_omic6
            )           

        risk = -torch.sum(S, dim=1).cpu().numpy()
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time

        pred_prob.append(torch.softmax(grad, dim=-1)[0].cpu().numpy())
        true_prob.append(label_g.cpu().numpy())
        pred_grad.append(int(torch.argmax(torch.softmax(grad, dim=-1), dim=-1)[0]))
        true_grad.append(int(torch.argmax(label_g, dim=-1)[0]))

        attention[slide_id] = A

        result.append({
            'slide_id': slide_id,
            'maginification_level': level,
            'true_grade': true_grad[-1],
            'pred_prob_grad': pred_prob[-1],
            'pred_grade': pred_grad[-1],
            'censorship': all_censorships[batch_idx],
            'event_time': all_event_times[batch_idx],
            'pred_risk': risk 
        })

    torch.save(attention, args.results_dir+'/'+args.data_type+'_attention.pt')

    df_results = pd.DataFrame(result)
    df_results.to_excel(args.results_dir + '/'+args.data_type+'_overall_prediction.xlsx', index=False)
    
    
    # Compute metrics for overall predictions
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    kappa = cohen_kappa_score(true_grad, pred_grad, weights='quadratic')
    balanced_acc = balanced_accuracy_score(true_grad, pred_grad)
    f1 = f1_score(true_grad, pred_grad, average='micro')
    fpr['overall'], tpr['overall'], _ = roc_curve(np.array(true_prob).ravel(), np.array(pred_prob).ravel(), pos_label=1)
    roc_auc['overall'] = auc(fpr['overall'], tpr['overall']) 
    aucc = roc_auc['overall']

    # Save and print results
    message = f"Overall -> \n C Index: {c_index} -- AUC: {aucc} -- Kappa: {kappa} -- F1: {f1} -- Balanced Acc: {balanced_acc}"
    print(message)

    with open(writer, "a") as log_file:
        log_file.write('%s\n' % message)



def test_distributed(args, model, loader, writer=None, results_dir=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Storage for slide ids and grades
    results, pred_grade, true_grade = [], [], []
    pred_grad, true_grad, pred_prob, true_prob, result = [], [], [], [], []
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (slide_id, level, data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label_g, label_s, event_time, c) in tqdm(enumerate(loader)):
        data_WSI = data_WSI.to(device)
        data_omic1 = data_omic1.type(torch.FloatTensor).to(device)
        data_omic2 = data_omic2.type(torch.FloatTensor).to(device)
        data_omic3 = data_omic3.type(torch.FloatTensor).to(device)
        data_omic4 = data_omic4.type(torch.FloatTensor).to(device)
        data_omic5 = data_omic5.type(torch.FloatTensor).to(device)
        data_omic6 = data_omic6.type(torch.FloatTensor).to(device)
        label_g = label_g.type(torch.FloatTensor).to(device)
        label_s = label_s.type(torch.LongTensor).to(device)
        c = c.type(torch.FloatTensor).to(device)

        # Split data_WSI into M equal groups
        num_embeddings = data_WSI.size(1)
        M = num_embeddings
        group_size = num_embeddings // M
        groups = [data_WSI[:, i * group_size:(i + 1) * group_size, :] for i in range(M)]

        group_grades, group_risks, group_grad_prob = [], [], []
        for group in groups:
            with torch.no_grad():
                grad, surv, hazards, S, Y_hat, A = model(
                    x_path=group,
                    x_omic1=data_omic1, 
                    x_omic2=data_omic2, 
                    x_omic3=data_omic3, 
                    x_omic4=data_omic4, 
                    x_omic5=data_omic5, 
                    x_omic6=data_omic6
                ) 
                group_grades.append(int(torch.argmax(torch.softmax(grad, dim=-1), dim=-1)[0]))
                group_risks.append(-torch.sum(S, dim=1).cpu().numpy())
                group_grad_prob.append(torch.softmax(grad, dim=-1)[0].cpu().numpy())


        results.append({
            "slide_id": slide_id,
            "grades": group_grades,
            'risks': group_risks,
            'level': level
        })

        all_risk_scores[batch_idx] = sum(group_risks)/len(group_risks)
        all_censorships[batch_idx] = c.cpu().numpy()
        all_event_times[batch_idx] = event_time

        pred_prob.append(np.mean(np.array(group_grad_prob), axis=0))
        true_prob.append(label_g.cpu().numpy())
        pred_grad.append(int(round(sum(group_grades)/len(group_grades))))
        true_grad.append(int(torch.argmax(label_g, dim=-1)[0]))

        result.append({
            'slide_id': slide_id,
            'maginification_level': level,
            'true_grade': true_grad[-1],
            'pred_prob_grad': pred_prob[-1],
            'pred_grade': pred_grad[-1],
            'censorship': all_censorships[batch_idx],
            'event_time': all_event_times[batch_idx],
            'pred_risk': all_risk_scores[batch_idx]
        })

    # Save results to a .pt file
    torch.save(results, args.results_dir+'/'+args.data_type+'_distributed.pt')
    print('\nGrades saved at %s.'%(args.results_dir))

    df_results = pd.DataFrame(result)
    df_results.to_excel(args.results_dir + '/'+args.data_type+'_distributed_prediction.xlsx', index=False)

    # Compute metrics for overall predictions
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    kappa = cohen_kappa_score(true_grad, pred_grad, weights='quadratic')
    balanced_acc = balanced_accuracy_score(true_grad, pred_grad)
    f1 = f1_score(true_grad, pred_grad, average='micro')
    fpr['overall'], tpr['overall'], _ = roc_curve(np.array(true_prob).ravel(), np.array(pred_prob).ravel(), pos_label=1)
    roc_auc['overall'] = auc(fpr['overall'], tpr['overall']) 
    aucc = roc_auc['overall']

    # Save and print results
    message = f"Distributed -> \n C Index: {c_index} -- AUC: {aucc} -- Kappa: {kappa} -- F1: {f1} -- Balanced Acc: {balanced_acc}"
    print(message)

    with open(writer, "a") as log_file:
        log_file.write('%s\n' % message)



