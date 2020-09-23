# example layout for DCC computing

# packages
library(tidyverse)
library(cowplot)

# run data from CSV
WESAD_data <- function(directory = "/hpc/group/sta440-f20/rsh33/imputed_data/", n) {
  cur <- paste0(directory, "S", n, ".csv", sep="")
  return(read_csv(cur))
}

S2 <- WESAD_data(n=2)

# analysis
hist_plot <- function(person, variable, cuts) {
  
  d = get(paste0("S",person,sep=""))
  
  p1 <- d[variable] %>%
    ggplot(aes(x="", y = eval(as.name(variable)))) +
    geom_boxplot(fill = "lightblue", color = "black") + 
    coord_flip() +
    theme_classic() +
    xlab("") + ylab(variable) +
    theme(axis.text.y=element_blank(),
          axis.ticks.y=element_blank())
  
  p2 <- d[variable] %>%
    ggplot() +
    geom_histogram(aes(x = eval(as.name(variable))), bins = cuts, 
                   fill = "lightblue") +
    ylab("Count") + xlab("") +
    theme_classic()
  
  cowplot::plot_grid(p2, p1,
                     ncol = 1, rel_heights = c(2, 1),
                     align = 'v', axis = 'lr') 
} 

p = 2
v = 100

# output 
hist_plot(person=p, variable="EMG", cuts=v)