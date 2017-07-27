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

#str(data)
#levels(data$fake_news_category_2)
#unique(data$is_fake_news_2)
#table(data$is_fake_news_2)

library(ggplot2)

# plot distribution of fake news by user_verified
ggplot(data, aes(x=as.integer(user_verified), fill=fake)) +
  geom_histogram(position='dodge', aes(y=(..density..)/sum(..density..)),  binwidth=0.5) +
  #scale_fill_manual(values = c('#9fea7e', '#ea877e'), name="fake news") +
  xlab("user verified") + 
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
  #xlim(0, 5000000) +
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
  geom_histogram(position='dodge', aes(y=(..density..)/sum(..density..))) +
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
  geom_histogram(position='dodge', aes(y=(..density..)/sum(..density..))) +
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
  geom_histogram(position='dodge', aes(y=..density../sum(..density..)), binwidth=0.5) +
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
ggplot(data, aes(x=num_media, fill=fake, colour=fake)) +
  geom_histogram(position='dodge', aes(y=(..density..)/sum(..density..))) +
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
