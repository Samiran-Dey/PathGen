import torch
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

scaler_list = joblib.load('Data/TCGA_KIRC/org_minmax_scaler.pkl')

device = "cuda" if torch.cuda.is_available() else "cpu"
    

class MyDataset(Dataset):
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        files = [os.listdir(folder_path[i]) for i in range(len(folder_path))]
        self.file_list = sorted([folder_path[i]+item for i in range(len(files)) for item in files[i]])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        with open(file_path, 'rb') as file:
            emb_dict = torch.load(file, weights_only=False)

        slide_id = emb_dict['slide_id']
        level = emb_dict['meta_data']['level']
        WSI_embedding = emb_dict['WSI_embedding']

        omic_data_Tumor_Suppressor_Genes = emb_dict['omic_data_Tumor Suppressor Genes'][0]
        omic_data_Oncogenes = emb_dict['omic_data_Oncogenes'][0]
        omic_data_Protein_Kinases = emb_dict['omic_data_Protein Kinases'][0]
        omic_data_Cell_Differentiation_Markers = emb_dict['omic_data_Cell Differentiation Markers'][0]
        omic_data_Transcription_Factors = emb_dict['omic_data_Transcription Factors'][0]
        omic_data_Cytokines_and_Growth_Factors = emb_dict['omic_data_Cytokines and Growth Factors'][0]

        return (
            slide_id,
            level,
            WSI_embedding,
            omic_data_Tumor_Suppressor_Genes,
            omic_data_Oncogenes,
            omic_data_Protein_Kinases,
            omic_data_Cell_Differentiation_Markers,
            omic_data_Transcription_Factors,
            omic_data_Cytokines_and_Growth_Factors
        )

def create_dataset(folder_path, batch_size=1):
    dataset = MyDataset(folder_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
