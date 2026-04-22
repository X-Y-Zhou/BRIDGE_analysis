import numpy as np
import matplotlib.pyplot as plt
import velocyto as vcy
import protaccel.protaccel as pa

[prot_count_array,prot_cells,adt_names] = pa.import_prot_data('8309714/GSM3596101_CTCL-ADT-count.csv')
vlm = vcy.VelocytoLoom("8309714/190426_CTCL.loom")

vlm.score_cv_vs_mean(3000, plot=True, max_expr_avg=100, min_expr_cells=20)
vlm.score_detection_levels(min_expr_counts=40, min_cells_express=30)
# plt.show()

# U0 = vlm.U
# S0 = vlm.S
# gene_names_all = vlm.ra["Gene"]

gene_dict = {'B220 (CD45R)':['PTPRC'], 'B7-H1 (PD-L1)':['CD274'],  'C-kit (CD117)':['KIT'], 'CCR7':['CCR7'], 'CD11b':['ITGAM'], 'CD11c':['ITGAX'], 'CD138':['SDC1'], 'CD14':['CD14'], 'CD16':['FCGR3A'], 'CD19':['CD19'], 'CD1a':['CD1A'], 'CD2':['CD2'], 'CD223 (lag3)':['LAG3'],  'CD26 (Adenosine)':['DPP4'], 'CD27':['CD27'], 'CD28':['CD28'], 'CD3':['CD3E'], 'CD34':['CD34'], 'CD366 (tim3)':['HAVCR2'], 'CD4':['CD4'], 'CD44':['CD44'], 'CD45':['PTPRC'], 'CD45RA':['PTPRC'], 'CD45RO':['PTPRC'], 'CD5':['CD5'], 'CD56':['NCAM1'], 'CD62L':['SELL'], 'CD66b':['CEACAM8'], 'CD69':['CD69'], 'CD7':['CD7'],  'CD8':['CD8A'], 'CTLA4':['CTLA4'], 'EpCAM (CD326)':['EPCAM'], 'HLA-A,B,C':['HLA-A'], 'IL7Ralpha (CD127)':['IL7R'],   'LAMP1':['LAMP1'], 'MHCII (HLA-DR)':['HLA-DRA'], 'Ox40 (CD134)':['TNFRSF4'], 'PD-1 (CD279)':['PDCD1'], 'PD-L1 (CD274)':['CD274'], 'PD1 (CD279)':['PDCD1'],  'Siglec-8':['SIGLEC8']}
InvertDict = lambda d: dict( (v,k) for k in d for v in d[k] )
prot_dict = InvertDict(gene_dict)
mrna_targets = list(prot_dict.keys())
pa.enforce_protein_filter(vlm,mrna_targets,adt_names)

# velocyto.py的package中内置的函数对所有的基因进行了一次筛选
vlm.filter_genes(by_cv_vs_mean=True,by_detection_levels=True)

# vlm.filter_genes 不仅仅对基因进行简单筛选，还可以对基因表达矩阵进行了一些操作，因为筛选后的基因表达矩阵结果不同，并不是进行抽取
# U1 = vlm.U
# S1 = vlm.S
# gene_names = vlm.ra["Gene"]
# shared_gene_names = np.asarray([np.where(gene_names_all==cell)[0][0] for cell in gene_names])

# sum(sum(U0[shared_gene_names,:] == U1)) 
# sum(sum(S0[shared_gene_names,:] == S1))

first_char = vlm.ca['CellID'][0].find(':')+1
last_char = -1

[prot_count_array, shared_cells, prot_cells] = pa.shared_cells_filter(vlm, prot_cells, prot_count_array, first_char, last_char)

pa.impute(vlm, prot_count_array, k=800, impute_in_prot_space=True, size_norm=False, impute_in_pca_space=False)

t_cl =  [0,0,1,2,1]
[cluster_ID, num_clusters] = pa.identify_clusters(vlm,vlm.connectivity,
                                                  correct_tags=True,tag_correction_list=t_cl,
                                                  method_name='ModularityVertexPartition')

COLORS=np.asarray([[0, 0.4470, 0.7410],
        [0.4660, 0.6740, 0.1880], [0.9290, 0.6940, 0.1250]])
cluster_labels = ['CD4+ T','CD8+ T','Mono.']
vlm.COLORS = COLORS
vlm.labels=cluster_labels

pa.fit_pcs(vlm,'P_norm','prot_pcs',n_pcs=4)
pa.visualize_pcs(vlm, [1,2])
plt.show()

pa.visualize_pcs(vlm, [2,3])
plt.show()


list(vlm.ra.keys())
gene_names = vlm.ra["Gene"]
CellIDs = vlm.ca['CellID']

adt_names
prot_count_array
vlm.U
vlm.S

import pandas as pd
df = pd.DataFrame(vlm.U, index=gene_names, columns=CellIDs)
df.to_csv("8309714/data_filtered_CTCL/U2.csv")

df = pd.DataFrame(vlm.S, index=gene_names, columns=CellIDs)
df.to_csv("8309714/data_filtered_CTCL/S2.csv")

df = pd.DataFrame(prot_count_array, index=adt_names, columns=CellIDs)
df.to_csv("8309714/data_filtered_CTCL/P2.csv")


cluster_ID
len(cluster_ID)
np.savetxt("8309714/data_filtered_CTCL/cluster_ID.txt",cluster_ID,fmt="%.0f",delimiter="\t")


# vlm
# prot_cells
# prot_count_array
# first_char
# last_char
# filter_empty_cells=True
# min_cell_size=5


# if last_char == 0:
#         rna_cells = np.asarray([cell_str[first_char:] for cell_str in vlm.ca['CellID']])
# else:
#         rna_cells = np.asarray([cell_str[first_char:last_char] for cell_str in vlm.ca['CellID']])
#         shared_cells = [cell_tag for cell_tag in rna_cells if cell_tag in prot_cells]

# print('ADT cell number: '+str(len(prot_cells)))
# print('RNAseq cell number: '+str(len(rna_cells)))
# print('Shared cells: '+str(len(shared_cells)))

# shared_cell_prot_ind = np.asarray([np.where(prot_cells==cell)[0][0] for cell in shared_cells])
# shared_cell_rna_ind = np.asarray([np.where(rna_cells==cell)[0][0] for cell in shared_cells])

# prot_count_array[:,shared_cell_prot_ind]
# shared_cell_prot_ind
# shared_cell_rna_ind

# prot_count_array = prot_count_array.astype('float')

# if filter_empty_cells:
#         non_sparse_cell_filt = (vlm.U[:,shared_cell_rna_ind].sum(0)>min_cell_size)  \
#             & (vlm.S[:,shared_cell_rna_ind].sum(0)>min_cell_size)  \
#             & (prot_count_array[:,shared_cell_prot_ind].sum(0)>min_cell_size)
#         print('Shared cells with more than '+str(min_cell_size)+' molecules: '+str(sum(non_sparse_cell_filt)))

#         shared_cell_prot_ind = shared_cell_prot_ind[non_sparse_cell_filt]
#         shared_cell_rna_ind = shared_cell_rna_ind[non_sparse_cell_filt]

# for col_att in vlm.ca:
#         vlm.ca[col_att] = vlm.ca[col_att][shared_cell_rna_ind]

# np.savetxt("8309714/data_filtered_CTCL/shared_cell_rna_ind.txt",shared_cell_rna_ind,fmt="%.0f",delimiter="\t")

# shared_cell_rna_ind
# vlm.S
# vlm.U

# vlm.S[:,shared_cell_rna_ind]
# vlm.U[:,shared_cell_rna_ind]


# vlm.S = vlm.S[:,shared_cell_rna_ind]
# vlm.U = vlm.U[:,shared_cell_rna_ind]
# prot_count_array = prot_count_array[:,shared_cell_prot_ind]
# prot_cells = np.asarray(shared_cells) #for idempotence

# vlm.S
# prot_count_array

# vlm.P = prot_count_array
    

# [prot_count_array, shared_cells, prot_cells]