library(tidyr)
library(stringr)
library(tidytext)
library(dplyr)
library(drlib)
library(ggplot2)
library(stm)

debates_raw <- read_csv("hansard_test.csv", col_names = TRUE)

debates <- as_tibble(debates_raw)

debates


tidy_debates <- debates %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words) %>%
  filter(word != "hon") %>%
  filter(word != "government") %>%
  filter(word != "minister") %>%
  filter(word != "chancellor") %>%
  filter(word != "friend") %>%
  filter(word != "billion") %>%
  filter(word != "million")
  
tidy_debates

tidy_debates %>%
  count(word, sort = TRUE)

debates_tf_idf <- tidy_debates %>%
  count(id, word, sort = TRUE) %>%
  bind_tf_idf(word, id, n) %>%
  arrange(-tf_idf) %>%
  group_by(id) %>%
  top_n(10) %>%
  ungroup

debates_tf_idf

debates_tf_idf %>%
  mutate(word = reorder_within(word, tf_idf, id)) %>%
  ggplot(aes(word, tf_idf, fill = id)) +
  geom_col(alpha = 0.8, show.legend = FALSE) +
  facet_wrap(~ id, scales = "free", ncol = 3) +
  scale_x_reordered() +
  coord_flip() +
  theme(strip.text=element_text(size=11)) +
  labs(x = NULL, y = "tf-idf",
       title = "Highest tf-idf words in debates",
       subtitle = "Individual debates focus on different topics")

debates_dfm <- tidy_debates %>%
  count(id, word, sort = TRUE) %>%
  cast_dfm(id, word, n)

debates_sparse <- tidy_debates %>%
  count(id, word, sort = TRUE) %>%
  cast_sparse(id, word, n)

topic_model <- stm(debates_sparse, K = 6, 
                   verbose = FALSE, init.type = "Spectral")

td_beta <- tidy(topic_model)

td_beta %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  mutate(topic = paste0("Topic ", topic),
         term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(term, beta, fill = as.factor(topic))) +
  geom_col(alpha = 0.8, show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free_y") +
  coord_flip() +
  scale_x_reordered() +
  labs(x = NULL, y = expression(beta),
       title = "Highest word probabilities for each topic",
       subtitle = "Different words are associated with different topics")

td_gamma <- tidy(topic_model, matrix = "gamma",                    
                 document_names = rownames(debates_dfm))

ggplot(td_gamma, aes(gamma, fill = as.factor(topic))) +
  geom_histogram(alpha = 0.8, show.legend = FALSE) +
  facet_wrap(~ topic, ncol = 3) +
  labs(title = "Distribution of document probabilities for each topic",
       subtitle = "Each topic is associated with 1-3 debates",
       y = "Number of debates", x = expression(gamma))
