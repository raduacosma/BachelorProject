#
#     Copyright (C) 2021  Radu Alexandru Cosma
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

# Some specific plot commands that were used for the plots
# in the thesis

t0<-read.csv("SimpleFinal/000.txt")
t0<-read.csv("SimpleFinal/001.txt")
t0<-read.csv("SimpleFinal/002.txt")
t0<-read.csv("SimpleFinal/000.txt")
t1<-read.csv("SimpleFinal/001.txt")
t2<-read.csv("SimpleFinal/002.txt")
plotAll(t0%>% slice(seq(1,10000,1)),'opPredPerc','single MLP for all opponents (SMA)',t1 %>% slice(seq(1,10000,1)),'opPredPerc','loss distribution comparison (LDC)',t2%>% slice(seq(1,10000,1)),'opPredPerc','change-point detection (CPD)','Episode Number','Mean Opponent Move Prediction (%)',' Correctly predicted moves (%), Sarsa, Simple game','opponent modelling method')

plotAll(t0%>% slice(seq(1,10000,1)),'opPredPerc','single MLP for all opponents (SMA)',t1 %>% slice(seq(1,10000,1)),'foundOpPredPerc','loss distribution comparison (LDC)',t2%>% slice(seq(1,10000,1)),'foundOpPredPerc','change-point detection (CPD)','Episode Number','Mean Opponent Move Prediction (%)',' Correctly predicted moves (%) when found, Sarsa, Simple game','opponent modeling method')




t0<-read.csv('ComplexFinal/000.txt')
t1<-read.csv('ComplexFinal/012.txt')
t2<-read.csv('ComplexFinal/021.txt')


plotAll(t0%>% slice(seq(1,10000,1)),'opPredPerc','Sarsa (SAR), single MLP for all opponents (SMA)',t1 %>% slice(seq(1,10000,1)),'foundOpPredPerc','Q-learning (QL), change-point detection (CPD)',t2%>% slice(seq(1,10000,1)),'foundOpPredPerc','Double QL (DQL), loss distribution comparison (LDC)','Episode Number','Mean Opponent Move Prediction (%)',' Correctly predicted moves (%) when found, Complex game','algorithm combination')
plotAll(t0%>% slice(seq(1,10000,1)),'rewards','Sarsa (SAR), simple MLP for all opponents (SMA)',t1 %>% slice(seq(1,10000,1)),'rewards','Q-Learning (QL), change-point detection (CPD)',t2%>% slice(seq(1,10000,1)),'rewards','Double QL (DQL), loss distribution comparison (LDC)','Episode Number','Mean Reward per Episode',' Mean reward per episode over all agents, Complex game','algorithm combination')


t0<-read.csv('ComplexFinal/102.txt')
t1<-read.csv('ComplexFinal/112.txt')
t2<-read.csv('ComplexFinal/122.txt')


plotAll(t0%>% slice(seq(1,10000,1)),'rewards','Sarsa (SAR), change-point detection (CPD)',t1 %>% slice(seq(1,10000,1)),'rewards','Q-Learning (QL), change-point detection (CPD)',t2%>% slice(seq(1,10000,1)),'rewards','Double QL (DQL), change-point detection (CPD)','Episode Number','Mean Reward per Episode',' Mean reward per episode over all agents, Complex game, Random','algorithm combination')
 
t0<-read.csv('ComplexFinal/100.txt')
t1<-read.csv('ComplexFinal/101.txt')
t2<-read.csv('ComplexFinal/102.txt')
plotAll(t0%>% slice(seq(1,10000,1)),'opPredPerc','Sarsa (SAR), single MLP for all opponents (SMA)',t1 %>% slice(seq(1,10000,1)),'foundOpPredPerc','Sarsa (SAR), loss distribution comparison (LDC)',t2%>% slice(seq(1,10000,1)),'foundOpPredPerc','Sarsa (SAR), change-point detection (CPD)','Episode Number','Mean Opponent Move Prediction (%)',' Correctly predicted moves (%) when found, Complex game, Random','algorithm combination')