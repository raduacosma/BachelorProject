folderName = "SimpleFinal"
plotType = "rewards"

t0<-read.csv("SimpleFinal/001.txt")
t1<-read.csv("SimpleFinal/002.txt")
t2<-read.csv("SimpleFinal/003.txt")





plotAll(t0%>% slice(seq(1,10000,1)),plotType,'oneforall',
        t1 %>% slice(seq(1,10000,1)),plotType,'kols',t2%>% slice(seq(1,10000,1)),
        plotType,'pettitt','Episode Number','Reward',
        'Reward per episode, complex game, best of all',
        'opponent modeling method')

