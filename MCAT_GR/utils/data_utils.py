
import torch
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

def minmax_normalize(tensor):
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
    return normalized_tensor 

class MyDataset(Dataset):
    def __init__(self, folder_path, data_path = None, syn=True, test_type='overall', real_ratio=None, noise=False, data_filter=None):
        super().__init__()
        self.folder_path = folder_path
        self.syn = syn
        self.noise=noise
        files = [os.listdir(folder_path[i]) for i in range(len(folder_path))]
        all_files = [folder_path[i]+item for i in range(len(files)) for item in files[i]]
        self.test_type = test_type
        if data_path:
            self.gen = torch.load(data_path, map_location=torch.device('cpu'), weights_only=False)
           
        self.real_ratio = real_ratio
        if real_ratio is not None:
            num_indices = int(len(self.file_list) * real_ratio)
            all_indices = list(range(len(self.file_list)))
            self.real_indices = random.sample(all_indices, num_indices)

        if data_filter is None: 
            self.file_list = all_files
        else:
            self.file_list = []
            for file_path in all_files:
                with open(file_path, 'rb') as file:
                    emb_dict = torch.load(file, weights_only=False)
                
                include_file = True

                # Filter by grade
                if data_filter.get('grade'):
                    if int(emb_dict['meta_data']['grade']) != data_filter['grade']:
                        include_file = False

                # Filter by is_female
                if data_filter.get('is_female') is not None:
                    if emb_dict['meta_data'].get('is_female') != data_filter['is_female']:
                        include_file = False

                # Filter by age range
                if data_filter.get('age'):
                    age = emb_dict['meta_data'].get('age')
                    age_min, age_max = data_filter['age']
                    if age_min is not None and age < age_min:
                        include_file = False
                    if age_max is not None and age >= age_max:
                        include_file = False

                # Filter by magnification level
                if data_filter.get('mag_level'):
                    if int(emb_dict['meta_data']['level']) != data_filter['mag_level']:
                        include_file = False

                # Filter by censorship
                if data_filter.get('censorship') is not None:
                    if emb_dict.get('censorship') != data_filter['censorship']:
                        include_file = False

                # Filter by risk level
                if data_filter.get('time_bin') is not None:
                    if emb_dict['meta_data'].get('time_bin') != data_filter['time_bin']:
                        include_file = False

                if include_file:
                    self.file_list.append(file_path)    

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):

        file_path = self.file_list[index]
        with open(file_path, 'rb') as file:
            emb_dict = torch.load(file, weights_only=False)


        # Extract necessary components from emb_dict
        slide_id = emb_dict['slide_id']
        WSI_embedding = emb_dict['WSI_embedding']
        k = (slide_id,)

        meta_data = emb_dict['meta_data']
        survival_months = meta_data['survival_months']
        censorship = emb_dict['censorship']
        level = meta_data['level']
        label_grade = torch.tensor(emb_dict['label'])
        label_surv = torch.tensor(meta_data['time_bin'])

        if self.real_ratio is not None:
            if index in self.real_indices:
                omic_data_Tumor_Suppressor_Genes = minmax_normalize(emb_dict['omic_data_Tumor Suppressor Genes'][0])
                omic_data_Oncogenes = minmax_normalize(emb_dict['omic_data_Oncogenes'][0])
                omic_data_Protein_Kinases = minmax_normalize(emb_dict['omic_data_Protein Kinases'][0])
                omic_data_Cell_Differentiation_Markers = minmax_normalize(emb_dict['omic_data_Cell Differentiation Markers'][0])
                omic_data_Transcription_Factors = minmax_normalize(emb_dict['omic_data_Transcription Factors'][0])
                omic_data_Cytokines_and_Growth_Factors = minmax_normalize(emb_dict['omic_data_Cytokines and Growth Factors'][0])
            else:
                omic_data_Tumor_Suppressor_Genes = minmax_normalize(self.gen[k][0])
                omic_data_Oncogenes = minmax_normalize(self.gen[k][1])
                omic_data_Protein_Kinases = minmax_normalize(self.gen[k][2])
                omic_data_Cell_Differentiation_Markers = minmax_normalize(self.gen[k][3])
                omic_data_Transcription_Factors = minmax_normalize(self.gen[k][4])
                omic_data_Cytokines_and_Growth_Factors = minmax_normalize(self.gen[k][5])
        else: 
            if self.syn:
                    omic_data_Tumor_Suppressor_Genes = minmax_normalize(self.gen[k][0])
                    omic_data_Oncogenes = minmax_normalize(self.gen[k][1])
                    omic_data_Protein_Kinases = minmax_normalize(self.gen[k][2])
                    omic_data_Cell_Differentiation_Markers = minmax_normalize(self.gen[k][3])
                    omic_data_Transcription_Factors = minmax_normalize(self.gen[k][4])
                    omic_data_Cytokines_and_Growth_Factors = minmax_normalize(self.gen[k][5])
                                   
            else:
                omic_data_Tumor_Suppressor_Genes = minmax_normalize(emb_dict['omic_data_Tumor Suppressor Genes'][0])
                omic_data_Oncogenes = minmax_normalize(emb_dict['omic_data_Oncogenes'][0])
                omic_data_Protein_Kinases = minmax_normalize(emb_dict['omic_data_Protein Kinases'][0])
                omic_data_Cell_Differentiation_Markers = minmax_normalize(emb_dict['omic_data_Cell Differentiation Markers'][0])
                omic_data_Transcription_Factors = minmax_normalize(emb_dict['omic_data_Transcription Factors'][0])
                omic_data_Cytokines_and_Growth_Factors = minmax_normalize(emb_dict['omic_data_Cytokines and Growth Factors'][0])


        if self.noise:
            omic_data_Tumor_Suppressor_Genes = minmax_normalize(torch.randn_like(omic_data_Tumor_Suppressor_Genes))
            omic_data_Oncogenes = minmax_normalize(torch.randn_like(omic_data_Oncogenes))
            omic_data_Protein_Kinases = minmax_normalize(torch.randn_like(omic_data_Protein_Kinases))
            omic_data_Cell_Differentiation_Markers = minmax_normalize(torch.randn_like(omic_data_Cell_Differentiation_Markers))
            omic_data_Transcription_Factors = minmax_normalize(torch.randn_like(omic_data_Transcription_Factors))
            omic_data_Cytokines_and_Growth_Factors = minmax_normalize(torch.randn_like(omic_data_Cytokines_and_Growth_Factors))


        return (
            slide_id,
            level,
            WSI_embedding,
            omic_data_Tumor_Suppressor_Genes,
            omic_data_Oncogenes,
            omic_data_Protein_Kinases,
            omic_data_Cell_Differentiation_Markers,
            omic_data_Transcription_Factors,
            omic_data_Cytokines_and_Growth_Factors,
            label_grade,
            label_surv,
            survival_months,
            censorship
        )

def create_dataset(args, folder_path, op_mode=None, batch_size=1, data_filter=None):
    if op_mode is None:
        op_mode = args.op_mode

    if op_mode == 'test' and args.data_type == 'merged': 
        print('Loading synthesized and real data ...')
        dataset = MyDataset(folder_path, args.test_syn_path, False, args.real_ratio, data_filter=data_filter)
        print('\nreal_ratio: ', args.real_ratio)
    elif op_mode == 'calibrate' and args.data_type == 'syn':
        print('Loading synthesized ...')
        if args.test_type == 'overall':
            dataset = MyDataset(folder_path, args.cal_syn_path, syn=True, test_type=args.test_type, data_filter=data_filter)

    elif op_mode == 'test' and args.data_type == 'syn':
        print('Loading synthesized ...')
        dataset = MyDataset(folder_path, args.test_syn_path, syn=True, test_type=args.test_type, data_filter=data_filter)
    elif args.data_type == 'noise':
        print('Loading without genomic ...')
        dataset = MyDataset(folder_path, syn=False, noise=True, data_filter=data_filter)
    else:
        print('Loading real data ...')
        dataset = MyDataset(folder_path, syn=False, data_filter=data_filter)
   
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)



