################################################################################
################################################################################
# File: ###################  Event Study regressions  #######################
################################################################################
################################################################################

# Author: Clinton Leung, Gian-Piero Lovicu, David Ampudia Vicente
# Last Update: 03-30-22

################################################################################
#### Package and directory Control Panel ####
################################################################################

# Clean out the workspace
rm(list = ls())
memory.limit(72000)
options(max.print=1000)

# Check installation & LOAD PACKAGES 
list.of.packages <- c("tidyverse", "ggplot2", "fixest", "broom", "fastDummies", "RColorBrewer")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, repos = "http://cran.us.r-project.org")

invisible(lapply(list.of.packages, library, character.only = TRUE))


setwd(dirname(rstudioapi::getSourceEditorContext()$path))
setwd("..")

figs <- "figs/"
tables <- "tables/"
data_proc <- "data/processed_data/"
data_simul <- "data/simul_results/"

figs <- list()


select <- dplyr::select
summarise <- dplyr::summarise

################################################################################
#### Data Loading & Pre-Processing ####
################################################################################
baseline_stats <- read.csv(paste(data_proc, "summary_stats.csv", sep = "")) %>%
  distinct(name, .keep_all = TRUE) %>%
  select(-(c(tie)|contains("caught")|contains("eigen_prop")))

df <- data.frame()
files <-str_subset(list.files("data/simul_results"), ".csv")

for (i in files) {
  toy_data <- read.csv(paste(data_simul, i, sep=""), skip=2) %>%
    `colnames<-`(c(
      "strat", 
      "name", 
      "sim", 
      "caught_0.25", 
      "caught_0.50",
      "caught_0.75",
      "caught_1.00", 
      "eigen_0.25", 
      "eigen_0.50",
      "eigen_0.75",
      "eigen_1.00", 
      "unfinished"
    ))
  f <- plyr::join(toy_data, baseline_stats, by="name", type="left")
  df <- rbind(df, f)
}
