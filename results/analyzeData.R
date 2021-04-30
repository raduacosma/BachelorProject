library(ggplot2)
library(data.table)

qtotal<-read.csv("QLearning_totalRewards.txt")
dqtotal<-read.csv("DQLearning_totalRewards.txt")
satotal<-read.csv("Sarsa_totalRewards.txt")
esatotal<-read.csv("ExpectedSarsa_totalRewards.txt")

qtotal=qtotal/10000;

dqtotal=dqtotal/10000;

satotal=satotal/10000;

esatotal=esatotal/10000;

mean(qtotal$means)
sd(qtotal$means)

mean(dqtotal$means)
sd(dqtotal$means)

mean(satotal$means)
sd(satotal$means)

mean(esatotal$means)
sd(esatotal$means)

esaq <- t.test(esatotal,qtotal)
esaq
p.adjust(esaq$p.value, method = "bonferroni", 3)

esadq <- t.test(esatotal,dqtotal)
esadq
p.adjust(esadq$p.value, method = "bonferroni", 3)

esasa<-t.test(esatotal,satotal)
esasa
p.adjust(esasa$p.value, method = "bonferroni", 3)



