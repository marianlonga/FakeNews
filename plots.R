# read data and adjust data types
data <- read.csv(file='twitter_data_OneMonth_6000.csv')
data$created_at <- as.character(data$created_at)
data$text <- as.character(data$text)
data$user_screen_name <- as.character(data$user_screen_name)
#data$is_fake_news_2 <- as.factor(data$is_fake_news_2)
data$is_fake_news_2 <- as.character(data$is_fake_news_2)
data$fake_news_category_2 <- as.factor(data$fake_news_category_2)
data$user_verified <- as.factor(data$user_verified)

# add 'fake' column to specify user readable label when displayed on historgram ("fake news"/"other")
data$fake = data$is_fake_news_2
data$fake[data$fake == "TRUE"] <- "fake news"
data$fake[data$fake == "FALSE"] <- "other"
# add 'fake' column to specify if tweet is fake news (TRUE/FALSE)
#data$fake = data$is_fake_news_2

# drop rows where is_fake_news_2 is unknown ("UNKNOWN" / "UNKOWN") or hasn't been filled out ("") 
data <- data[! data$fake %in% c("", "UNKNOWN", "UNKOWN"),]
#data <- droplevels(data)

data_fake  = data[data$fake == "fake news", ]
data_other = data[data$fake == "other", ]

#str(data)
#levels(data$fake_news_category_2)
#unique(data$is_fake_news_2)
#table(data$is_fake_news_2)

library(ggplot2)

# plot distribution of fake news by user_verified
ggplot(data, aes(x=as.integer(user_verified), fill=fake)) +
  geom_histogram(position='dodge', aes(y=(..density..)/sum(..density..)),  binwidth=1) +
  #scale_fill_manual(values = c('#9fea7e', '#ea877e'), name="fake news") +
  xlab("user verified (1=no, 2=yes)") + 
  ylab("normalised density of tweets") +
  ggtitle("Distribution of fake news by verified/non-verified user") +
  theme(
    legend.position = c(0.05, 0.95), 
    legend.title = element_blank(),
    legend.justification = c('left', 'top')
  )
ggsave("plots/1_user_verified.png")

#ggplot(data, aes(x=user_verified, fill=fake)) +
#  geom_bar() +
#  #scale_fill_manual(values = c('#9fea7e', '#ea877e'), name="fake news") +
#  coord_polar(theta='y') +
#  #xlab("user verified") + 
#  #ylab("number of tweets") +
#  ggtitle("Distribution of fake news by verified/non-verified user") +
#  theme(
#    legend.position = c(0.95, 0.95), 
#    legend.title = element_blank(),
#    legend.justification = c('right', 'top')
#  )

# distribution of fake news by retweet count
ggplot(data, aes(x=log10(retweet_count), fill=fake, colour=fake)) +
  geom_density(alpha=0.5) +
  xlim(2.5, 4.5) +
  xlab("log10(number of retweets)") + 
  ylab("density") +
  ggtitle("Distribution of fake news by retweet count") +
  theme(
    legend.position = c(0.95, 0.95), 
    legend.title = element_blank(),
    legend.justification = c('right', 'top')
  )
ggsave("plots/2_retweet_count.png")

# distribution of fake news by number of friends
ggplot(data, aes(x=log10(user_friends_count), fill=fake, colour=fake)) +
  geom_density(alpha=0.5) +
  #xlim(0, 5000) +
  xlab("log10(number of friends)") + 
  ylab("density") +
  ggtitle("Distribution of fake news by number of friends") +
  theme(
    legend.position = c(0.95, 0.95), 
    legend.title = element_blank(),
    legend.justification = c('right', 'top')
  )
ggsave("plots/3_friends_count.png")

# distribution of fake news by number of followers
ggplot(data, aes(x=log10(user_followers_count), fill=fake, colour=fake)) +
  geom_density(alpha=0.5) +
  xlim(2, 8) +
  xlab("log10(number of followers)") + 
  ylab("density") +
  ggtitle("Distribution of fake news by number of followers") +
  theme(
    legend.position = c(0.95, 0.95), 
    legend.title = element_blank(),
    legend.justification = c('right', 'top')
  )
ggsave("plots/4_followers_count.png")

# distribution of fake news by number of favourites
ggplot(data, aes(x=log10(user_favourites_count), fill=fake, colour=fake)) +
  geom_density(alpha=0.5) +
  #xlim(0, 5000000) +
  xlab("log10(number of favourites)") + 
  ylab("density") +
  ggtitle("Distribution of fake news by number of favourites") +
  theme(
    legend.position = c(0.95, 0.95), 
    legend.title = element_blank(),
    legend.justification = c('right', 'top')
  )
ggsave("plots/5_favourites_count.png")

# distribution of fake news by number of hashtags
ggplot(data, aes(x=num_hashtags, fill=fake, colour=fake)) +
  geom_histogram(position='dodge', aes(y=..density../sum(..density..)), binwidth=1) +
  #geom_histogram(position='dodge', binwidth=1) +
  #geom_bar() +
  #xlim(0, 5000000) +
  xlab("number of hashtags") + 
  ylab("normalised density of tweets") +
  ggtitle("Distribution of fake news by number of hashtags") +
  theme(
    legend.position = c(0.95, 0.95), 
    legend.title = element_blank(),
    legend.justification = c('right', 'top')
  )
ggsave("plots/6_num_hashtags.png")

# distribution of fake news by number of mentions
ggplot(data, aes(x=num_mentions, fill=fake, colour=fake)) +
  geom_histogram(position='dodge', aes(y=(..density..)/sum(..density..)), binwidth=1) +
  #geom_bar() +
  #xlim(0, 5000000) +
  xlab("number of mentions") + 
  ylab("normalised density of tweets") +
  ggtitle("Distribution of fake news by number of mentions") +
  theme(
    legend.position = c(0.95, 0.95), 
    legend.title = element_blank(),
    legend.justification = c('right', 'top')
  )
ggsave("plots/7_num_mentions.png")

# distribution of fake news by number of urls
ggplot(data, aes(x=num_urls, fill=fake, colour=fake)) +
  geom_histogram(position='dodge', aes(y=..density../sum(..density..)), binwidth=1) +
  #geom_bar() +
  #xlim(0, 5000000) +
  xlab("number of urls") + 
  ylab("normalised density of tweets") +
  ggtitle("Distribution of fake news by number of urls") +
  theme(
    legend.position = c(0.95, 0.95), 
    legend.title = element_blank(),
    legend.justification = c('right', 'top')
  )
ggsave("plots/8_num_urls.png")

# distribution of fake news by number of media
ggplot(data, aes(x=num_media, fill=fake)) +
  geom_histogram(position='dodge', aes(y=(..density..)/sum(..density..)), binwidth=1) +
  #geom_bar() +
  #xlim(0, 5000000) +
  xlab("number of media") + 
  ylab("normalised density of tweets") +
  ggtitle("Distribution of fake news by number of media") +
  theme(
    legend.position = c(0.95, 0.95), 
    legend.title = element_blank(),
    legend.justification = c('right', 'top')
  )
ggsave("plots/9_num_media.png")



## distribution of fake news by number of digits in username ##

library(stringr)
number_of_digits <- function(text) {
  sum <- 0
  sum <- sum + str_count(text, '0')
  sum <- sum + str_count(text, '1')
  sum <- sum + str_count(text, '2')
  sum <- sum + str_count(text, '3')
  sum <- sum + str_count(text, '4')
  sum <- sum + str_count(text, '5')
  sum <- sum + str_count(text, '6')
  sum <- sum + str_count(text, '7')
  sum <- sum + str_count(text, '8')
  sum <- sum + str_count(text, '9')
  return(sum)
}
#username_contains_digits = grepl('\\d',data$user_screen_name)
ggplot(data, aes(x=number_of_digits(user_screen_name), fill=fake)) +
  geom_histogram(position='dodge', aes(y=(..density..)/sum(..density..)),  binwidth=0.5) +
  xlab("number of digits in username") + 
  ylab("normalised density of tweets") +
  ggtitle("Distribution of fake news by number of digits in username") +
  theme(
    legend.position = c(0.95, 0.95), 
    legend.title = element_blank(),
    legend.justification = c('right', 'top')
  )
ggsave("plots/10_username_number_of_digits.png")



## number of underscores in a username ##

ggplot(data, aes(x=str_count(user_screen_name, '_'), fill=fake)) +
  geom_histogram(position='dodge', aes(y=(..density..)/sum(..density..)),  binwidth=0.5) +
  xlab("number of underscores in username") + 
  ylab("normalised density of tweets") +
  ggtitle("Distribution of fake news by number of underscores in username") +
  theme(
    legend.position = c(0.95, 0.95), 
    legend.title = element_blank(),
    legend.justification = c('right', 'top')
  )
ggsave("plots/11_username_number_of_underscores.png")



## number of capital letters in a username ##
ggplot(data, aes(x=str_count(user_screen_name, '[A-Z]'), fill=fake)) +
  geom_histogram(position='dodge', aes(y=(..density..)/sum(..density..)),  binwidth=0.5) +
  xlab("number of capital letters in username") + 
  ylab("normalised density of tweets") +
  ggtitle("Distribution of fake news by number of capital letters in username") +
  theme(
    legend.position = c(0.95, 0.95), 
    legend.title = element_blank(),
    legend.justification = c('right', 'top')
  )
ggsave("plots/12_username_number_of_caps.png")




# generate t statistics and p values

ttest <- t.test(as.integer(data_fake$user_verified), as.integer(data_other$user_verified))
stats1 <- c("user_verified", ttest$statistic, ttest$p.value)
ttest <- t.test(data_fake$retweet_count, data_other$retweet_count)
stats2 <- c("retweet_count", ttest$statistic, ttest$p.value)
ttest <- t.test(data_fake$user_friends_count, data_other$user_friends_count)
stats3 <- c("user_friends_count", ttest$statistic, ttest$p.value)
ttest <- t.test(data_fake$user_followers_count, data_other$user_followers_count)
stats4 <- c("user_followers_count", ttest$statistic, ttest$p.value)
ttest <- t.test(data_fake$user_favourites_count, data_other$user_favourites_count)
stats5 <- c("user_favourites_count", ttest$statistic, ttest$p.value)
ttest <- t.test(data_fake$num_hashtags, data_other$num_hashtags)
stats6 <- c("num_hashtags", ttest$statistic, ttest$p.value)
ttest <- t.test(data_fake$num_mentions, data_other$num_mentions)
stats7 <- c("num_mentions", ttest$statistic, ttest$p.value)
ttest <- t.test(data_fake$num_urls, data_other$num_urls)
stats8 <- c("num_urls", ttest$statistic, ttest$p.value)
ttest <- t.test(data_fake$num_media, data_other$num_media)
stats9 <- c("num_media", ttest$statistic, ttest$p.value)

ttest <- t.test(number_of_digits(data_fake$user_screen_name), number_of_digits(data_other$user_screen_name))
stats10 <- c("number_of_digits in user_screen_name", ttest$statistic, ttest$p.value)
ttest <- t.test(str_count(data_fake$user_screen_name, '_'), str_count(data_other$user_screen_name, '_'))
stats11 <- c("number_of_underscores in user_screen_name", ttest$statistic, ttest$p.value)
ttest <- t.test(str_count(data_fake$user_screen_name, '[A-Z]'), str_count(data_other$user_screen_name, '[A-Z]'))
stats12 <- c("number_of_caps in user_screen_name", ttest$statistic, ttest$p.value)

stats1
stats2
stats3
stats4
stats5
stats6
stats7
stats8
stats9
stats10
stats11
stats12
