library(ggplot2)
library(data.table)
library(gridExtra)
library(grid)
library(zoo)

rewards = read.table("rewards.txt")
setDT(rewards);
rewards$id = 1:nrow(rewards);
