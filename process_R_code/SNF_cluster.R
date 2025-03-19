setwd("../data")

library('SNFtool')
library("ConsensusClusterPlus")
library("survival")
library("survminer")

get_snf_cluster <- function(cancer,cluster_type,gene_num) {
  
  # driver gene
  driver_gene <- read.table("reference/CGC.txt")[,1]
  
  # exp_data
  exp_data <- read.table(paste0("TCGA_UCSC_EXP/",cancer,"/",cancer,"_exp_data.txt"))
  exp_sample <- read.table(paste0("TCGA_UCSC_EXP/",cancer,"/",cancer,"_sample.txt"))[,1]
  exp_gene <- read.table(paste0("TCGA_UCSC_EXP/",cancer,"/",cancer,"_gene.txt"))[,1]
  colnames(exp_data) <- exp_sample
  rownames(exp_data) <- exp_gene
  exp_data <- exp_data[intersect(exp_gene,driver_gene),]
  
  # met_data
  met_data <- read.table(paste0("TCGA_UCSC_MET/",cancer,"/",cancer,"_met_data.txt"),sep = "\t")
  met_sample <- read.table(paste0("TCGA_UCSC_MET/",cancer,"/",cancer,"_sample.txt"))[,1]
  met_gene <- read.table(paste0("TCGA_UCSC_MET/",cancer,"/",cancer,"_gene.txt"))[,1]
  colnames(met_data) <- met_sample 
  rownames(met_data) <- met_gene
  met_data <- met_data[intersect(met_gene,driver_gene),]
  
  # filter gene
  exp_gene_var <- apply(exp_data,1,var)
  met_gene_var <- apply(met_data,1,var)
  exp_first_var <- exp_gene_var[order(exp_gene_var,decreasing = T)]
  met_first_var <- met_gene_var[order(met_gene_var,decreasing = T)]
  
  #SNF
  dataX <- t(exp_data[names(exp_first_var)[1:gene_num],])
  dataY <- t(met_data[names(met_first_var)[1:gene_num],])
  K <- 20
  alpha <- 0.5
  T <- 20
  truelabel <- c(matrix(1,nrow(dataX),1),matrix(1,nrow(dataY),1))
  
  dataX <- standardNormalization(dataX)
  dataY <- standardNormalization(dataY)
  Dist1 <- (dist2(as.matrix(dataX),as.matrix(dataX)))^(1/2)
  Dist2 <- (dist2(as.matrix(dataY),as.matrix(dataY)))^(1/2)
  W1 = affinityMatrix(Dist1, K, alpha)
  W2 = affinityMatrix(Dist2, K, alpha)
  # displayClusters(W1,truelabel);
  # displayClusters(W2,truelabel);
  W = SNF(list(W1,W2), K, T)
  
  # cluster number
  cluster_best <- estimateNumberOfClustersGivenGraph(W, NUMC=3:9)
  print(cluster_best)
  if (cluster_type == 0) {
    C = max(c(cluster_best[[1]],cluster_best[[2]]))
  } else {
    C = cluster_best[[cluster_type]]
  }
  
  # spectralClustering
  res_group <- spectralClustering(W,C,type = 3)
  displayClusters(W,res_group)
  res_group <- as.matrix(res_group)
  colnames(res_group) <- "cluster"
  rownames(res_group) <- rownames(dataX)
  
  return(res_group)
}


cancer_types <- c("TCGA-BLCA", "TCGA-BRCA", "TCGA-COAD", "TCGA-HNSC", 
                  "TCGA-KIRC", "TCGA-KIRP", "TCGA-LIHC", "TCGA-LUAD", 
                  "TCGA-LUSC", "TCGA-PRAD", "TCGA-THCA", "TCGA-UCEC")

# snf clustering
for (cancer in cancer_types) 
{
  res_group <- get_snf_cluster(cancer,1,100)
  cluster_num <- max(res_group[,1])
  
  # survival data
  clin <- read.csv(paste0("survival/",cancer,".survival.tsv"),sep = "\t")
  clin <- clin[which(duplicated.data.frame(clin)==FALSE),]
  rownames(clin) <- clin$sample
  dat <- clin
  dat$group = 0
  for (k in 1:cluster_num) {
    group_k <- rownames(res_group)[which(res_group[,1] == k)]
    dat[group_k,"group"] <- k
  }
  
  # survival
  dat <- dat[which(dat$group != 0 & is.na(dat$sample) == F),]
  my.surv <- Surv(dat$OS.time,dat$OS==1)
  kmfit2 <- survfit(my.surv~dat$group,data=dat)
  ggsurvplot(kmfit2,conf.int = F,pval=T,risk.table = F,ncensor.plot=F)
  
  # write cluster label
  res_group <- cbind(res_group,rownames(res_group))
  colnames(res_group) <- c("cluster","sample")
  table(res_group[,1])
  write.table(res_group,paste0("survival/subtype_cluster/",cancer,"_cluster.txt"),
              sep = "\t",quote = F,col.names = F,row.names = F)
  write.table(table(res_group[,1]),paste0("survival/subtype_cluster/",cancer,"_table.txt"),
              sep = "\t",quote = F,col.names = T,row.names = F)
}



