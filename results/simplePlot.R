library(ggplot2)
library(data.table)
library(gridExtra)
library(grid)
library(zoo)


rewards = read.table("opponentFirstEpLoss.txt")
setDT(rewards)
rewards$id = 1:nrow(rewards);
ggplot(rewards,(aes(x=id,y=V1)))+geom_line()+ggtitle("Q-Learning in a simple game")+
  theme(plot.title = element_text(hjust = 0.5, size = 16)) +
  labs(x="Episode Number",y="Total Reward Per Episode")
theme(axis.title.x = element_text(size=16)) +
  theme(axis.title.y = element_text(size=16))+
  theme(axis.text=element_text(size=15))