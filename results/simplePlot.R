library(ggplot2)
library(data.table)
library(gridExtra)
library(grid)
library(zoo)

rewards = read.table("rewardsSimpleWall.txt")
setDT(rewards)
rewards$id = 1:nrow(rewards);
ggplot(rewards,(aes(x=id,y=V1)))+geom_line()
