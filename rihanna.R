#--------------------------------------------------------
# STEP 1 - Import 2000 Tweets
#--------------------------------------------------------

library(twitteR)


# Create variables for consumer_key, consume_secret, access_token, and access_secret

consumer_key='GIsEh1hH7szBlIj9yudSySHOs'
consumer_secret='YMbjH2D6h9jprY2mjsjI7sp5BQSF9HRLtSAYTQcjLYFDfYKQzK'
access_token='766387157959319552-qKYk8JBdMhD3SEH7VdHDo2WT7X4QMwQ'
access_secret='4U97p95ow3R8eUAnLCrEC6fFMyLmPejAu0FeXftctgr31'

setup_twitter_oauth(consumer_key, consumer_secret, access_token, access_secret)
tw = twitteR::searchTwitter("@rihanna", n = 2000, since = '2011-01-01',retryOnRateLimit = 1e3)
rihanna = twitteR::twListToDF(tw)
head(rihanna)
dim(rihanna)

# saving data to CSV
setwd("E:\\Data Analytics\\Phani\\Text Analytics\\")
write.csv(rihanna, file = "rihanna.csv")

rihanna <- read.csv('rihanna.csv')
rihanna


library(qdap)
library(tm)

#----------------------------------------------------------------------------
# WORD CLOUD ---------

rihanna$text <- iconv(rihanna$text, from = "UTF-8", to = "ASCII", sub = "")
rihanna_corp <- VCorpus(VectorSource(rihanna$text))

#Function to extract text and apply stemming and stopwords.
clean_corpus <- function(cleaned_corpus){
  removeURL <- content_transformer(function(x) gsub("(f|ht)tp(s?)://\\S+", "", x, perl=T))
  cleaned_corpus <- tm_map(cleaned_corpus, removeURL)
  cleaned_corpus <- tm_map(cleaned_corpus, content_transformer(replace_abbreviation))
  cleaned_corpus <- tm_map(cleaned_corpus, content_transformer(tolower))
  cleaned_corpus <- tm_map(cleaned_corpus, removePunctuation)
  cleaned_corpus <- tm_map(cleaned_corpus, removeNumbers)
  cleaned_corpus <- tm_map(cleaned_corpus, removeWords, stopwords("english"))
  # available stopwords
  # stopwords::stopwords() #can replace with custom stop words. Go biggest string to smallest. 
  custom_stop_words <- c("@rihanna", "rt")
  cleaned_corpus <- tm_map(cleaned_corpus, removeWords, custom_stop_words)
  # cleaned_corpus <- tm_map(cleaned_corpus, stemDocument,language = "english")
  cleaned_corpus <- tm_map(cleaned_corpus, stripWhitespace)
  return(cleaned_corpus)}

#Applying the function
cleaned_review_corpus <- clean_corpus(rihanna_corp)
head(cleaned_review_corpus)
print(cleaned_review_corpus[[1]][1])

#Convert to a term document matrix
TDM_rihanna <- TermDocumentMatrix(cleaned_review_corpus)
TDM_reviews_m <- as.matrix(TDM_rihanna)
TDM_reviews_m[1:10, 1:10]

# Term Frequency - plot
term_frequency <- rowSums(TDM_reviews_m)
# Sort term_frequency in descending order
term_frequency <- sort(term_frequency,dec=TRUE)
# View the top 20 most common words
top20 <- term_frequency[1:20]
# Plot a barchart of the 20 most common words
barplot(top20,col="darkorange",las=2)

# Word Cloud
library(wordcloud)
term_frequency[1:10]
# Create word_freqs
word_freqs <- data.frame(term = names(term_frequency), num = term_frequency)


#--------------------------------------------------------
# STEP 2 - CREATE A UNIGRAM, BIGRAM & TRIGRAM WORD CLOUD
#--------------------------------------------------------

#Unigram - Create a unigram wordcloud
wordcloud(word_freqs$term, word_freqs$num,min.freq=15,max.words=2000,colors=brewer.pal(8, "Paired"))

#Bigram -  
library(RWeka)
tokenizer <- function(x)
  NGramTokenizer(x,Weka_control(min=2,max=2))

bigram_tdm <- TermDocumentMatrix(cleaned_review_corpus,control = list(tokenize=tokenizer))
bigram_tdm_m <- as.matrix(bigram_tdm)
term_frequency <- rowSums(bigram_tdm_m)
# Sort bigram term_frequency in descending order
term_frequency <- sort(term_frequency,dec=TRUE)
# Create word_freqs
word_freqs <- data.frame(term = names(term_frequency), num = term_frequency)
#Make Bigram Wordcloud -  Create a wordcloud for the values in word_freqs
wordcloud(word_freqs$term, word_freqs$num,min.freq=15,max.words=2000,colors=brewer.pal(8, "Paired"))

#Trigram
tokenizer <- function(x)
  NGramTokenizer(x,Weka_control(min=3,max=3))

trigram_tdm <- TermDocumentMatrix(cleaned_review_corpus,control = list(tokenize=tokenizer))
trigram_tdm_m <- as.matrix(trigram_tdm)
term_frequency <- rowSums(trigram_tdm_m)
# Sort trigram term_frequency in descending order
term_frequency <- sort(term_frequency,dec=TRUE)
# Create word_freqs
word_freqs <- data.frame(term = names(term_frequency), num = term_frequency)
#Make trigram Wordcloud -  Create a wordcloud for the values in word_freqs
wordcloud(word_freqs$term, word_freqs$num,min.freq=15,max.words=2000,colors=brewer.pal(8, "Paired"))
#Could not fit many of the trigrams as was the case for unigrams and bigrams.Need to play with max/min word settings.
warnings() 


#--------------------------------------------------------
# STEP 3 - CREATE A TFIDF WORD CLOUD
#--------------------------------------------------------

# tf-idf weighting

tfidf_tdm <- TermDocumentMatrix(cleaned_review_corpus,control=list(weighting=weightTfIdf))
tfidf_tdm_m <- as.matrix(tfidf_tdm)

# Term Frequency
term_frequency <- rowSums(tfidf_tdm_m)
# Sort term_frequency in descending order
term_frequency <- sort(term_frequency,dec=TRUE)

# Create word_freqs
word_freqs <- data.frame(term = names(term_frequency), num = term_frequency)
# Create a wordcloud for the values in word_freqs
wordcloud(word_freqs$term, word_freqs$num,min.freq=10,max.words=2000,colors=brewer.pal(8, "Paired"))


#--------------------------------------------------------
# STEP 4 - SENTIMENT ANALYSIS with BING LEXICON
#--------------------------------------------------------

library(tm)
library(qdap)
library(tibble)
library(ggplot2)
library(RWeka)
library(wordcloud)
library(lubridate)
library(lexicon)
library(tidytext)
library(lubridate)
library(gutenbergr)
library(stringr)
library(dplyr)
library(radarchart)

#Applying qdap polarity to all of the texts. Note- not splitting texts by sentence, reading entire text 

#polarity(
#  text.var   = rihanna$text,
#  polarity.frame = custom_pol,
#  negators       = qdapDictionaries::negation.words,
#  amplifiers     = qdapDictionaries::amplification.words,
#  deamplifiers   = qdapDictionaries::deamplification.words 
#)

#rihanna$polarity <- rihanna$text.apply(polarity) # doesn't work... 

#Bringing in Bing lexicon. 
bing_lex <- get_sentiments("bing")
table(bing_lex$sentiment)
bing_lex$sentiment <- ifelse(bing_lex$sentiment=="negative", -1, 1)
colnames(bing_lex)= c("x","y")

#Redefine the corpus cleaning function to work with the lexicons:
clean_corpus <- function(cleaned_corpus){
  cleaned_corpus <- tm_map(cleaned_corpus, removeWords, stopwords("english"))
  cleaned_corpus <- tm_map(cleaned_corpus, stripWhitespace)
  return(cleaned_corpus)
}
review_corpus <- VCorpus(VectorSource(rihanna$text))
cleaned_review_corpus <- clean_corpus(review_corpus)
cleaned_review_corpus

#Make tidy matrix
tidy_mytext <- tidy(TermDocumentMatrix(cleaned_review_corpus))
View(tidy_mytext)

dim(tidy_mytext)

#Join the text and bing lex to see the polarities. :
bing_lex <- get_sentiments("bing")
mytext_bing <- inner_join(tidy_mytext, bing_lex, by = c("term" = "word"))
View(mytext_bing)

#Score the tweets. 
mytext_bing$sentiment_n <- ifelse(mytext_bing$sentiment=="negative", -1, 1)
mytext_bing$sentiment_score <- mytext_bing$count*mytext_bing$sentiment_n
mytext_bing$sentiment_score #individual scores.
dim(mytext_bing)

#get aggregate sentiment score per tweet. 
aggdata <- aggregate(mytext_bing$sentiment_score, list(index = mytext_bing$document), sum)
aggdata
sapply(aggdata,typeof)
aggdata$index <- as.numeric(aggdata$index)
aggdata

#Plot the sentiment over the 2000 tweets -
barplot(aggdata$x, names.arg = aggdata$index)
#Smooth plot
ggplot(aggdata, aes(index, x)) + geom_point()
ggplot(aggdata, aes(index, x)) + geom_smooth() + theme_bw()+
  geom_hline(yintercept = 0, color = "red")+xlab("sentence")+ylab("sentiment")+
  ggtitle("Rihanna Sentiment")
barplot(aggdata$x, names.arg = aggdata$index)

#--------------------------------------------------------
# STEP 5 - CONTRAST/COMPARISON CLOUDS BASED ON SENTIMENT
#--------------------------------------------------------

rihanna$polarity <- 0
counter=1
for (i in rihanna$text){
  rihanna$polarity[counter]<- polarity(i)$all$polarity
  counter=counter+1
}
rihanna$polarity

#Next breakup the text by positive and negative, preprocess and spit out answers. 
#Subset refresher in R.hgg

positive_tweet <- rihanna[rihanna$polarity>0, "text"]
negative_tweets <- rihanna[rihanna$polarity<0, "text"]

clean_corpus <- function(cleaned_corpus){
  removeURL <- content_transformer(function(x) gsub("(f|ht)tp(s?)://\\S+", "", x, perl=T))
  cleaned_corpus <- tm_map(cleaned_corpus, removeURL)
  cleaned_corpus <- tm_map(cleaned_corpus, content_transformer(replace_abbreviation))
  cleaned_corpus <- tm_map(cleaned_corpus, content_transformer(tolower))
  cleaned_corpus <- tm_map(cleaned_corpus, removePunctuation)
  cleaned_corpus <- tm_map(cleaned_corpus, removeNumbers)
  cleaned_corpus <- tm_map(cleaned_corpus, removeWords, stopwords("english"))
  # available stopwords
  # stopwords::stopwords() #can replace with custom stop words. Go biggest string to smallest. 
  custom_stop_words <- c("@rihanna","rt")
  cleaned_corpus <- tm_map(cleaned_corpus, removeWords, custom_stop_words)
  # cleaned_corpus <- tm_map(cleaned_corpus, stemDocument,language = "english")
  cleaned_corpus <- tm_map(cleaned_corpus, stripWhitespace)
  return(cleaned_corpus)
}

positive_tweet

# Put positive and negative tweets next to each other.
positive_tweet<-paste(positive_tweet, collapse=" ")
negative_tweets <- paste(negative_tweets, collapse=" ")
pos_neg_tweets <- c(positive_tweet, negative_tweets)

pos_neg_tweets <- iconv(pos_neg_tweets, from = "UTF-8", to = "ASCII", sub = "")
review_corpus <- VCorpus(VectorSource(pos_neg_tweets))
clean_corpus_pos_neg <- clean_corpus(review_corpus)
clean_corpus_pos_neg


TDM_speech <- TermDocumentMatrix(clean_corpus_pos_neg)
TDM_speech_m <- as.matrix(TDM_speech)
TDM_speech_m

#Commonality cloud. 
commonality.cloud(TDM_speech_m,colors=brewer.pal(8, "Dark2"),max.words = 1000)
#Comparison Cloud
comparison.cloud(TDM_speech_m,colors=brewer.pal(8, "Dark2"),max.words = 500)


#--------------------------------------------------------
# STEP 6 - EMOTIONAL ANALYSIS - using NRC LEXICON
#--------------------------------------------------------

nrc_lex <- get_sentiments("nrc")
#Join the appropriate lexicon.
rihanna_nrc <- inner_join(tidy_mytext, nrc_lex, by = c("term" = "word"))
#Strip out pos and neg to only have emotions left.
rihanna_nrc_noposneg <- rihanna_nrc[!(rihanna_nrc$sentiment %in% c("positive","negative")),]
#Sum
aggdata <- aggregate(rihanna_nrc_noposneg$count, list(index = rihanna_nrc_noposneg$sentiment), sum)
#Plot
chartJSRadar(aggdata)

