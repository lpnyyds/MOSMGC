import h5py
import numpy as np
import sys
import argparse


def create_h5(cancer, data_path, out_path, network):
    f = h5py.File("{}{}_{}.h5".format(out_path, cancer, network), "w")
    filename = ["{}gene_feature/{}/{}_feature_{}.txt".format(data_path, cancer, cancer, network[0:-4]),
                "{}gene_feature/{}/{}_feature_name_{}.txt".format(data_path, cancer, cancer, network[0:-4]),
                "{}gene_feature/{}/{}_gene_{}.txt".format(data_path, cancer, cancer, network[0:-4]),
                "{}TCGA_UCSC_EXP/{}/{}_exp_data.txt".format(data_path, cancer, cancer),
                "{}TCGA_UCSC_EXP/{}/{}_sample.txt".format(data_path, cancer, cancer),
                "{}network/{}.txt".format(data_path, network),
                "{}reference/2179_non_driver_genes.txt".format(data_path),
                "{}TCGA_Mutation/{}/{}_mutation.txt".format(data_path, cancer, cancer),
                "{}TCGA_UCSC_normal/{}/{}_exp_normal.txt".format(data_path, cancer, cancer),
                "{}survival/subtype_cluster/{}_cluster.txt".format(data_path, cancer),
                "{}reference/CGC.txt".format(data_path)
                ]

    gene_feature = np.genfromtxt(filename[0], delimiter="\t", dtype=np.float32)
    feature_name = np.genfromtxt(filename[1], delimiter="\t", dtype=h5py.special_dtype(vlen=str))
    feature_name = feature_name[:, np.newaxis]
    gene_name = np.genfromtxt(filename[2], delimiter="\t", dtype=h5py.special_dtype(vlen=str))
    gene_name = gene_name[:, np.newaxis]
    exp_data = np.genfromtxt(filename[3], delimiter="\t", dtype=np.float32)
    sample_id = np.genfromtxt(filename[4], delimiter="\t", dtype=h5py.special_dtype(vlen=str))
    sample_id = sample_id[:, np.newaxis]
    ppi_network = np.genfromtxt(filename[5], delimiter="\t", dtype=h5py.special_dtype(vlen=str), skip_header=1)
    non_driver_gene = np.genfromtxt(filename[6], delimiter="\t", dtype=h5py.special_dtype(vlen=str))
    mut_matrix = np.genfromtxt(filename[7], delimiter="\t", dtype=np.int32)
    exp_normal = np.genfromtxt(filename[8], delimiter="\t", dtype=np.float32)
    sample_cluster = np.genfromtxt(filename[9], delimiter="\t", dtype=h5py.special_dtype(vlen=str))
    cgc = np.genfromtxt(filename[10], delimiter="\t", dtype=h5py.special_dtype(vlen=str))

    f.create_dataset("Gene_Feature", data=gene_feature)
    f.create_dataset("Feature_Name", data=feature_name)
    f.create_dataset("Gene_Name", data=gene_name)
    f.create_dataset("Exp_Data", data=exp_data)
    f.create_dataset("Sample_ID", data=sample_id)
    f.create_dataset("Network", data=ppi_network)
    f.create_dataset("Non_Driver_Gene", data=non_driver_gene)
    f.create_dataset("Mutation_Matrix", data=mut_matrix)
    f.create_dataset("Exp_Normal", data=exp_normal)
    f.create_dataset("Sample_Cluster", data=sample_cluster)
    f.create_dataset("CGC", data=cgc)
    f.close()
    return 1


parser = argparse.ArgumentParser()
parser.add_argument('--c', type=str, help='Cancer type, eg. TCGA-BRCA')
parser.add_argument('--n', type=str, help='PPI network, eg. STRING_v12')

if __name__ == "__main__":
    args = parser.parse_args()
    cancer = args.c
    print(cancer)
    network = args.n
    cancer_type = ["TCGA-BRCA", "TCGA-HNSC", "TCGA-BLCA", "TCGA-LUAD", "TCGA-UCEC",
                   "TCGA-KIRC", "TCGA-KIRP",  #
                   "TCGA-THCA", "TCGA-PRAD", "TCGA-COAD", "TCGA-LIHC", "TCGA-LUSC"]
    if not cancer in cancer_type:
        print("A cancer type needed: ")
        print(cancer_type)
        print("For example: python create_h5.py --c TCGA-BLCA --n string_full_v12_0.7")
        print("Exit now.")
        sys.exit(-1)

    data_path = "D:/Working/py_code/IHGC_2024_4/data/"
    out_path = "D:/Working/py_code/IHGC_2024_4/h5_file/"
    create_h5(cancer, data_path, out_path, network)
    print("Saved in {}{}_{}.h5".format(out_path, cancer, network))
