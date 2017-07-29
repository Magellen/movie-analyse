rm(list=ls())
setwd("D:/大四下/夏令营/作业题目/三.电影评论分析")
library(tidyverse)
library(stringr)
library(ggplot2)
library(jiebaR)
library(text2vec)
library(glmnet)
data<-read.csv("douban.csv")
data[,2]<-data[,2]%>%as.character()
for (i in 1:length(data[,2]))
{
  data[i,2]<-str_sub(data[i,2],2,5)
}
data[,2]%>%is.na()%>%sum
which(is.na(data[,2]))
data1<-data[c("上映年份","评分","评价人数")]
qplot(data1[,1],data1[,2])
qplot(data1[,1],data1[,3])
hero<-data%>%select(主演)%>%as.data.frame()
hero<-apply(hero,1,as.character)
herodist<-list[""]
herodist<-strsplit(hero,"/")
for (i in 1:1810)
{
  for(j in 1:length(herodist[[i]]))
  {
    herodist[[i]][j]<-gsub(" ","",herodist[[i]][j])
  }
}
mtype<-data%>%select(类型)%>%as.data.frame()
mtype<-apply(mtype,1,as.character)
mtypedist<-list[""]
mtypedist<-strsplit(mtype,"\\n")
mtypedist%>%unlist%>%unique()%>%length()
mtypem<-data.frame()
typename<-mtypedist%>%unlist%>%unique()
for(i in 1:1810)
{
  ifelse(sum(mtypedist[[i]] == typename),mtypem<-rbind(mtypem,(mtypedist[[i]] == typename)%>%as.numeric()),
         mtypem<-rbind(mtypem,rep(0,32)))
}
colnames(mtypem)<-c(typename)
typesummary<-mtypedist%>%summary
typesummary<-typesummary[,1]%>%as.numeric()
which(typesummary==0)



mplot<-data%>%select(剧情简介)%>%as.data.frame()
mplot<-apply(mplot,1,as.character)
for(j in 1:length(mplot))
{
  mplot[j]<-str_sub(mplot[j],3,nchar(mplot[j]))
}
length(mplot)
fenci<-worker(bylines = TRUE,type = "mix",stop_word = "chinese_stopword.txt")
fencidist<-list[""]
fencidist<-segment(mplot,fenci)
lapply(fencidist, write.table, "test2.txt", append=TRUE)
#时间因素分析

dim(data1)# 1810部电影
qplot(data1[,1],data1[,2])#评分随时间
qplot(data1[,1],data1[,3])#评论数随时间
qplot(data1[,1],log(data1[,3]))
qplot(data1[,1],log(data1[,3]),colour=data1[,2])#评论数随时间 颜色为评分
data1%>%group_by(上映年份)%>%count()%>%plot()#国产电影数量指数增长
quantile<-data1%>%group_by(上映年份)%>%count()
data%>%filter(上映年份>=2017)%>%select(片名)#提前透露
qplot(data1[,2],log(data1[,3]))#评论数和评分无关
data2000<-data%>%filter(上映年份>=2000&上映年份<2017)
View(data2000)
sub2000<-data2000%>%select(上映年份,评分,评价人数)
qplot(sub2000[,1],sub2000[,2])
p<-ggplot(sub2000,aes(上映年份,log(评价人数)))
p+geom_jitter()
qplot(sub2000[,2],log(sub2000[,3]))
summary(sub2000)
sub2000[,1]<-sub2000[,1]%>%as.integer()
sub2000_1<-data2000%>%group_by(上映年份)%>%select(上映年份,评价人数)
ggplot(sub2000_1,aes(x=log(评价人数)))+geom_histogram()+facet_grid(~上映年份)
data2000$导演%>%length()
data2000$导演%>%unique()

#text to vector
it <- itoken(iterable = fencidist)
vocab <- create_vocabulary(it)
vectorizer <- vocab_vectorizer(vocab)
corpus <- create_corpus(it,vectorizer)
vocab$vocab
dtm <- corpus$get_dtm()
dim(dtm)
dtm[1:20,2500:2530]
dtm_tfidf <- TfIdf$new()
dtm_tfidf$fit(dtm)
dtm_tfidf$transform(dtm)
dtm_tfidfm<-dtm_tfidf$transform(dtm)


vocab = create_vocabulary(it, ngram = c(1L, 2L))
vocab = vocab %>% prune_vocabulary(term_count_min = 2)
bigram_vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it, bigram_vectorizer)

vocab <- create_vocabulary(it)
vocab <- prune_vocabulary(vocab, term_count_min = 2)
vectorizer <- vocab_vectorizer(vocab,skip_grams_window = 2L)
tcm <- create_tcm(it, vectorizer)
dtm
glove = GlobalVectors$new(word_vectors_size = 20, vocabulary = vocab, x_max = 100)
glove$fit(tcm, n_iter = 200)
word_vectors <- glove$get_word_vectors()
dim(word_vectors)
vec<-list[]
for(i in 1:length(fencidist))
{
  for(j in 1:length(fencidist[[i]]))
  {
    vec[[i]]<-cbind(vec[[i]],word_vectors[rownames(word_vectors)==fencidist[[i]][j],])
  }
}

#TF-IDF
dtm_tfidfm%>%str()
tfidfm<-as.matrix(dtm_tfidfm)%>%as.data.frame()
dim(tfidfm)
write_csv(tfidfm,"tfidfm.csv")
write_csv(mtypem,"typem.csv")

#LDA
lda_model = 
  LDA$new(n_topics = 50, vocabulary = vocab, 
          doc_topic_prior = 0.1, topic_word_prior = 0.01)
doc_topic_distr = 
  lda_model$fit_transform(dtm, n_iter = 100, convergence_tol = 0.01, 
                          check_convergence_every_n = 10)
