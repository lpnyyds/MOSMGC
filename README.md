#MOSMGC: driver gene identification based on the clustering of tumor expression subtypes and multi-graph convolutional networks

### This is the original repository for the MOSMGC paper. 

**Requirements**

Python3.8

```
torch>=2.0.1
numpy>=1.24.3
torchmetrics>=1.2.1
sklearn>=1.3.2
scipy>=1.10.1
```

### **Input**

## Protein Protein Interaction (PPI) network

MOSMGC uses the PPI network, STRINGv12, as the underlying network. This network file can be download from https://cn.string-db.org/cgi/download.
The file should be stored at ./data/network/string_full_v12.txt.

For example:
```
protein_1	protein_2	score
ARF5	RALGPS2	173
ARF5	FHDC1	154
ARF5	ATP6V1E1	151
ARF5	CYTH2	471
...
```

## Multi-omics Data
Gene expression, promoter methylation, and copy number variants (CNV) data of cancer patients sourced from TCGA were meticulously gathered via the UCSC Xena Browser: https://xenabrowser.net/datapages/.
Single-nucleotide variation (SNV) data were downloaded through the R package "TCGAbiolinks"

For example:
```
library('TCGAbiolinks')

projects <- sort(getGDCprojects()$project_id)
cancer_type <- "TCGA-****"    #  "TCGA-BLCA" "TCGA-BRCA" "TCGA-COAD"  "TCGA-HNSC" "TCGA-KIRC" "TCGA-KIRP" "TCGA-LIHC" "TCGA-LUAD" "TCGA-LUSC" "TCGA-PRAD" "TCGA-THCA" "TCGA-UCEC"  
Project_Summary <- getProjectSummary(cancer_type)

# download SNV data
query <- GDCquery(
  project = cancer_type,
  data.category = "Simple Nucleotide Variation",
  data.type = "Masked Somatic Mutation", 
  access = "open",
  legacy = F
)
GDCdownload(query)

save_path <- "D:/Lenovo/Working/TCGA_data/iHGC/TCGA_hg38_SNV/"
# dir.create(save_path)
GDCprepare(query, save = T,save.filename = paste0(save_path,cancer_type,"_hg38_SNV.Rdata"))
```

## All tab-delimited omics data need to be pre-processed and stored in the following format. Under a given cancer type, all omics data of tumor samples maintain the same row and column names.

# SNV data:
The files should be stored in folder:  ./data/TCGA_hg38_SNV/snv_matrix/, including a gene ID file, a sample ID file and a SNV matrix.

For example:

1. Gene ID of TCGA-BLCA named TCGA-BLCA_snv_gene.txt. The gene IDs represent the row names of corresponding SNV matrix.
```
EPHA2
USP24
SERBP1
GJA5
....

2. Sample ID of TCGA-BLCA named TCGA-BLCA_snv_sample.txt. The sample IDs represent the column names of corresponding SNV matrix.
...
TCGA-XF-AAMH-01A
TCGA-GD-A2C5-01A
TCGA-DK-AA6W-01A
TCGA-FD-A6TF-01A
...

3. SNV matrix of TCGA-BLCA named TCGA-BLCA_snv_matrix.txt.
...
1	0	0	1	1	...
0	0	1	0	1	...
1	1	0	0	0	...
0	1	0	1	0	...
...
```

# Gene expression, promoter methylation, and CNV data  should be stored separately in the folders in a similar manner.
For gene expression data:	./data/TCGA_UCSC_EXP/TCGA-BLCA/
For  promoter methylation data:		./data/TCGA_UCSC_MET/TCGA-BLCA/
For CNV data:	./data/TCGA_UCSC_CNV/cnv_matrix/

# Gene expression and promoter methylation data of normal samples
For TCGA-BLCA, the data should be stored in the folder: 	./data/TCGA_UCSC_normal/TCGA-BLCA/, including  a gene ID file, a sample ID file and two data matrices.
The files are in the same format as the SNV data.

## Clinical data
Clinical data can be downloaded via the UCSC Xena Browser: https://xenabrowser.net/datapages/.

For cancer type TCGA-BLCA, the clinical data can be stored at ./data/survival/TCGA-BLCA.survival.tsv.
```
sample	OS	_PATIENT	OS.time
TCGA-E7-A8O8-01A	0	TCGA-E7-A8O8	13
TCGA-GC-A4ZW-01A	0	TCGA-GC-A4ZW	15
TCGA-E7-A5KE-01A	0	TCGA-E7-A5KE	17
TCGA-4Z-AA80-01A	1	TCGA-4Z-AA80	19
...
```

### reference driver genes

This file is located at ./data/reference/CGC.txt.

### **Run**

For example:

```
1. cd ./process_R_code/
	Run ./SNF_cluster.R
	Run ./get_gene_feature.R

2. cd ./src/
	python create_h5.py --c TCGA-BLCA --n string_full_v12_0.7

3. cd ./src/
	python train_MOSMGC_cv.py -e 2000 -lr 0.001 -hd [64, 128] -lm 2 -wd 5e-4 -do 0.5 -ns 0.2 -d '../h5_file/TCGA-BLCA_string_full_v12_0.7.h5' -cv 10 -dr 'cgc'
```

