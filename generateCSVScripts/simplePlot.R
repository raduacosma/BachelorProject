library(ggplot2)
library(data.table)
library(gridExtra)
library(grid)
library(zoo)
library(tidyverse)
library(dplyr)
library(matrixStats)

rewards = read.table("opponentPredictionLossesTwoDOUBLE.txt")
plotRewards<- function(data,column){
  rewardsCopy = data[column] %>% drop_na();
  rewardsCopy <- tibble::rowid_to_column(rewardsCopy, "ID")
  print(rewardsCopy)
 # setDT(rewards)
#  rewards$id = 1:nrow(rewards);
  currPlot<-ggplot(rewardsCopy,(aes(x=ID,y=opPredPerc)))+geom_line()+ggtitle("Q-Learning in a simple game")+
    theme(plot.title = element_text(hjust = 0.5, size = 16)) +
    labs(x="Episode Number",y="Total Reward Per Episode")
  theme(axis.title.x = element_text(size=16)) +
    theme(axis.title.y = element_text(size=16))+
    theme(axis.text=element_text(size=15)) 
  return(currPlot)
}


plotAll<-function(df,name,legend1,df2,name2,legend2,df3,name3,legend3,xtext,ytext,titletext,legendtitletext)
{
  
  
  columns<-c(paste(name,'1',sep=""),paste(name,'2',sep=""),paste(name,'3',sep=""),paste(name,'4',sep=""),paste(name,'5',sep=""),paste(name,'6',sep=""),paste(name,'7',sep=""),paste(name,'8',sep=""))
  dfCopy<-df ;
  totalLength <-nrow(dfCopy)
  dfCopy <- tibble::rowid_to_column(dfCopy, "ID")
  dfCopy<-dfCopy %>% 
    mutate(Mean= rowMeans(.[columns]), stderr=rowSds(as.matrix(.[columns])))
  dfCopy['stderr']<-dfCopy['stderr']/sqrt(8)
  IDs = seq(1,nrow(dfCopy['Mean']),100)
  
  test=tibble(Mean=numeric())
  for(i in seq(1,totalLength,100))
  {
    test<-test%>%add_row(Mean = deframe(summarise(dfCopy['Mean']%>%slice(seq(i,i+100,1)),Average=mean(Mean,na.rm=T)))[1])
  }
  test<-rename(test,Mean=1)
  meanSample <- test %>%  
    mutate(Mean= as.numeric(Mean))
  #meanSample=dfCopy['Mean']%>%slice(IDs)

  
  
  columns2<-c(paste(name2,'1',sep=""),paste(name2,'2',sep=""),paste(name2,'3',sep=""),paste(name2,'4',sep=""),paste(name2,'5',sep=""),paste(name2,'6',sep=""),paste(name2,'7',sep=""),paste(name2,'8',sep=""))
  dfCopy2<-df2 ;
  dfCopy2 <- tibble::rowid_to_column(dfCopy2, "ID")
  dfCopy2<-dfCopy2 %>% 
    mutate(Mean= rowMeans(.[columns2]), stderr=rowSds(as.matrix(.[columns2])))
  dfCopy2['stderr']<-dfCopy2['stderr']/sqrt(8)

  test=tibble(Mean=numeric())
  for(i in seq(1,totalLength,100))
  {
    test<-test%>%add_row(Mean = deframe(summarise(dfCopy2['Mean']%>%slice(seq(i,i+100,1)),Average=mean(Mean,na.rm=T)))[1])
  }
  test<-rename(test,Mean=1)
  meanSample2 <- test %>%  
    mutate(Mean= as.numeric(Mean))
  
  #meanSample2=dfCopy2['Mean']%>%slice(IDs)

  
  
  columns3<-c(paste(name3,'1',sep=""),paste(name3,'2',sep=""),paste(name3,'3',sep=""),paste(name3,'4',sep=""),paste(name3,'5',sep=""),paste(name3,'6',sep=""),paste(name3,'7',sep=""),paste(name3,'8',sep=""))
  dfCopy3<-df3 ;
  dfCopy3 <- tibble::rowid_to_column(dfCopy3, "ID")
  dfCopy3<-dfCopy3 %>% 
    mutate(Mean= rowMeans(.[columns3]), stderr=rowSds(as.matrix(.[columns3])))
  dfCopy3['stderr']<-dfCopy3['stderr']/sqrt(8)
  
  test=tibble(Mean=numeric())
  for(i in seq(1,totalLength,100))
  {
    test<-test%>%add_row(Mean = deframe(summarise(dfCopy3['Mean']%>%slice(seq(i,i+100,1)),Average=mean(Mean,na.rm=T)))[1])
  }
  test<-rename(test,Mean=1)
  meanSample3 <- test %>%  
    mutate(Mean= as.numeric(Mean))
  #meanSample3=dfCopy3['Mean']%>%slice(IDs)
  
  
  
  slicedLength<-nrow(meanSample)
  
  g1<-dfCopy['Mean']%>%add_column(dfCopy['stderr'])%>%rowid_to_column("ID")
  g2<-dfCopy2['Mean']%>%add_column(dfCopy2['stderr'])%>%rowid_to_column("ID")
  g3<-dfCopy3['Mean']%>%add_column(dfCopy3['stderr'])%>%rowid_to_column("ID")
  print(summarise(dfCopy['Mean']%>%slice(seq(7500,10000,1)),Average=mean(Mean,na.rm=T)))
  print(summarise(dfCopy2['Mean']%>%slice(seq(7500,10000,1)),Average=mean(Mean,na.rm=T)))
  print(summarise(dfCopy3['Mean']%>%slice(seq(7500,10000,1)),Average=mean(Mean,na.rm=T)))

  label1 = rep(legend1,totalLength)
  g1<-g1%>%add_column(label1)
  label1=rep(legend2,totalLength)
  g2<-g2%>%add_column(label1)
  label1=rep(legend3,totalLength)
  g3<-g3%>%add_column(label1)

  gTotal<-g1%>%full_join(g2)%>%full_join(g3)
  meanSample <- meanSample%>%add_column(IDs)
  lsliced = rep(legend1,slicedLength)
  meanSample<-meanSample%>%add_column(lsliced)
  
  meanSample2 <- meanSample2%>%add_column(IDs)
  lsliced = rep(legend2,slicedLength)
  meanSample2 <- meanSample2%>%add_column(lsliced)
  
  meanSample3 <- meanSample3%>%add_column(IDs)
  lsliced = rep(legend3,slicedLength)
  meanSample3 <- meanSample3%>%add_column(lsliced)
  meanSampleTotal<-meanSample%>%full_join(meanSample2)%>%full_join(meanSample3)
  currPlot<-ggplot()+
    geom_ribbon(data=gTotal,aes(x=ID,ymin=Mean-stderr,ymax=Mean+stderr,fill=label1),alpha=I(0.4))+
    
    # scale_fill_manual(values=alpha(c("red","green","blue"),.2))+
    #geom_ribbon(data=dfCopy,aes(x=ID,ymin=Mean-stderr,ymax=Mean+stderr),fill="red",alpha=0.2)+

    #geom_ribbon(data=dfCopy2,aes(x=ID,ymin=Mean-stderr,ymax=Mean+stderr),fill="blue",alpha=0.2)+

    #geom_ribbon(data=dfCopy3,aes(x=ID,ymin=Mean-stderr,ymax=Mean+stderr),fill="green",alpha=0.2)+
    geom_line(data=meanSampleTotal,aes(x=IDs,y=Mean,colour=lsliced),size=0.75)+
    labs(fill=legendtitletext,x=xtext,y=ytext)+
    theme()+
    guides(colour=FALSE,fill = guide_legend(override.aes = list(alpha=1)))+
    theme_bw()+
#    geom_line(data=meanSample,(aes(x=IDs,y=Mean)),colour="red2")+
#    geom_line(data=meanSample2,(aes(x=IDs,y=Mean)),colour="blue2")+
#    geom_line(data=meanSample3,(aes(x=IDs,y=Mean)),colour="green2")+
    ggtitle(titletext)+
    theme(legend.position = c(.55,.2),
          legend.title=element_text(size=22),
          legend.text=element_text(size=22),
          plot.title = element_text(hjust = 0.5, size = 22),
          axis.title.x = element_text(size=22),
          axis.title.y = element_text(size=22),
          axis.text=element_text(size=22))

  return(currPlot)
  
}