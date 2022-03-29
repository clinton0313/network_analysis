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
list.of.packages <- c("tidyverse", "ggplot2", "fixest", "broom", "fastDummies", "RColorBrewer", 
                      "modelsummary", "fixest", "wesanderson", "cowplot", "kableExtra", "scales")
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

df <- df %>%
  mutate(triang_p_node = triangles/nodes) %>%
  mutate(across(where(is.numeric), scale))

################################################################################
#### Model definition ####
################################################################################
covs <- list()
covs[["b0"]] <- c("degree_mean", "eig_cent_logratio")
covs[["b1"]] <- c(covs$b0, "eig_cent_mean", "density")
covs[["b2"]] <- c(covs$b1, "diameter", "triang_p_node", "average_clustering")
covs[["b3"]] <- c(covs$b2, 
                  "nodes", "edges", "degree_min", "eig_cent_max",
                  "eig_cent_min", "eig_cent_range"
                )

form.list <- list()
form.list[["c_m0"]] <- as.formula(
  paste0(
    "caught_0.25 ~", 
    paste0(covs$b0, collapse=" + "),
    "| strat"
  )
)

form.list[["c_m1"]] <- as.formula(
  paste0(
    "caught_0.25 ~", 
    paste0(covs$b1, collapse=" + "),
    "| strat"
  )
)

form.list[["c_m2"]] <- as.formula(
  paste0(
    "caught_0.25 ~", 
    paste0(covs$b2, collapse=" + "),
    "| strat"
  )
)

form.list[["c_m3"]] <- as.formula(
  paste0(
    "caught_0.25 ~", 
    paste0(covs$b3, collapse=" + "),
    "| strat"
  )
)

#### Eigenvectors ####

form.list[["e_m0"]] <- as.formula(
  paste0(
    "eigen_0.25 ~", 
    paste0(covs$b0, collapse=" + "),
    "| strat"
  )
)

form.list[["e_m1"]] <- as.formula(
  paste0(
    "eigen_0.25 ~", 
    paste0(covs$b1, collapse=" + "),
    "| strat"
  )
)

form.list[["e_m2"]] <- as.formula(
  paste0(
    "eigen_0.25 ~", 
    paste0(covs$b2, collapse=" + "),
    "| strat"
  )
)

form.list[["e_m3"]] <- as.formula(
  paste0(
    "eigen_0.25 ~", 
    paste0(covs$b3, collapse=" + "),
    "| strat"
  )
)

#### All Caught ####

form.list[["a_m0"]] <- as.formula(
  paste0(
    "eigen_1.00 ~", 
    paste0(covs$b0, collapse=" + "),
    "| strat"
  )
)

form.list[["a_m1"]] <- as.formula(
  paste0(
    "eigen_1.00 ~", 
    paste0(covs$b1, collapse=" + "),
    "| strat"
  )
)

form.list[["a_m2"]] <- as.formula(
  paste0(
    "eigen_1.00 ~", 
    paste0(covs$b2, collapse=" + "),
    "| strat"
  )
)

form.list[["a_m3"]] <- as.formula(
  paste0(
    "eigen_1.00 ~", 
    paste0(covs$b3, collapse=" + "),
    "| strat"
  )
)

################################################################################
#### Regressions ####
################################################################################

res0c <- feols(form.list$c_m0, data=df)
res1c <- feols(form.list$c_m1, data=df)
res2c <- feols(form.list$c_m2, data=df)
res3c <- feols(form.list$c_m3, data=df)

res0e <- feols(form.list$e_m0, data=df)
res1e <- feols(form.list$e_m1, data=df)
res2e <- feols(form.list$e_m2, data=df)
res3e <- feols(form.list$e_m3, data=df)

res2a <- feols(form.list$a_m2, data=df)
res3a <- feols(form.list$a_m3, data=df)

################################################################################
#### Plots ####
################################################################################

b <- list(geom_vline(xintercept = 0, color = 'orange'),
          annotate("rect", alpha = .1,
                   xmin = -1, xmax = 1.5, 
                   ymin = -Inf, ymax = Inf))

models <- list(
  "Caught Criminals" = res2c,
  "Caught Centrality" = res2e
)
coefplot1 <- modelplot(models, background=b, conf_level = .99) + 
  labs(x = "Coefficients", y = "Criminal group attributes",
       title = 'First Quantile') +
  scale_color_manual(values = wes_palette('Darjeeling1')) + 
  theme(legend.justification=c(0,0), legend.position=c(.55,.8))

coefplot2 <- modelplot(res2a, background=b, conf_level = .99) + 
  labs(x = "Coefficients", y = "Criminal group attributes",
       title = 'Last Quantile') +
  scale_color_manual(values = wes_palette('Darjeeling1'))

coefs <- cowplot::plot_grid(coefplot1, coefplot2, nrow = 1,
                  label_x = , label_y = , label_fontface = )

################################################################################
#### Tables ####
################################################################################

# function to get significance stars
make_stars <- function(t, dof) {
  if (2 * pt(-t, df=dof) < 0.01) {
    ptstar <- "***"
  } else if (2 * pt(-t, df=dof) < 0.05) {
    ptstar <- "**"
  } else if (2 * pt(-t, df=dof) < 0.1) {
    ptstar <- "*"
  } else {
    ptstar <- ""
  }
  return(ptstar)
}

# function to get info from models
get_info <- function(est, modelname, type) {
  bind_cols(
    broom::tidy(est, conf.int=TRUE) %>% 
      select(term, estimate, std.error, statistic, conf.low, conf.high),
    broom::glance(est) %>% 
      select(nobs, adj.r.squared, within.r.squared) %>%
      mutate(mod = modelname, type=type),
    num_id = length(unique(est$fixef_id$full_mun)),
    mdv = mean(est$fitted.values + est$residuals)
  )
}

output_table <- bind_rows(
  get_info(res2c, "First Quantile", "Caught Criminals"),
  get_info(res2e, "First Quantile", "Caught Centrality"),
  get_info(res2a, "Last Quantile", " ")
) %>% 
  rowwise() %>% 
  mutate(estimate = paste0(as.character(format(round(estimate, 2), nsmall = 2)), make_stars(statistic, 10000)),
   std.error = paste0("(", as.character(format(round(std.error, 2), nsmall = 2)), ")")) %>%
  select(term, estimate, std.error, type, mod) %>%
  gather(variable, value, -c(term,mod,type)) %>%
  pivot_wider(names_from=c(mod, type), values_from=value) %>%
  mutate(term = c("Mean Degree", "Logratio Eigenvector Cent.", "Eigenvector Cent. Mean",
                  "Density", "Diameter", "Triangles", "Mean Clustering", "",
                  "","","","","","")) %>%
  group_by(variable) %>%
  mutate(id_idx = row_number()) %>% 
  arrange(id_idx,variable) %>% 
  select(-id_idx) %>%
  kable("latex", align = 'lcccc', booktabs = T, linesep = c("", "","\\addlinespace"),
        col.names = NULL,
        escape=F,
        label = "table1", 
        caption = "The effect of network attribute on investigation effectiveness") %>% 
  kable_styling(position = "center", latex_options = c("HOLD_position")) %>%
  add_header_above(c("\\\\textit{Dependent Variable:}" = 1, "First Quantile" = 2, "Last Quantile"= 1), escape=F) %>%
  footnote(
    general_title = "",
    general = c(
      "\\\\footnotesize \\\\textit{Note:} The table reports results for the number of simulations it 
      takes to capture a fourth of a criminal network. The effects are in standard deviation units, 
      and strategies are included as fixed effects. \\\\ \\\\
      ***, **, and *
      indicate significance at the 0.01, 0.05 and 0.10 levels, respectively, using two-tailed tests. Robust standard errors are clustered
      at the municipality level."
    ),
    threeparttable = T ,
    footnote_as_chunk=T,
    escape = F
  )

write_lines(output_table, file = paste(tables, "table1.tex", sep = ""))
