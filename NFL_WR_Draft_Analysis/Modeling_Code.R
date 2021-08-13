# Load Libraries
library("rvest") 
library("caret")
library("ggplot2")
library("dplyr")
library("stringr")
library("kableExtra")
library("car")

# Create function to parse nfl pro football reference draft list html website data
parse_draft_class_future_nfl_stats <- function(start_year, end_year)
{
  final_df <- data.frame()
  for (year in start_year:end_year)
  {
    parser <- read_html(paste0("https://www.pro-football-reference.com/years/",year,"/draft.htm"))
    round <- parser %>% 
      html_nodes("th.right") %>%
      html_text() %>% as.numeric()
    text1 <- parser %>%
      html_nodes("tr") %>% 
      html_text()
    columns <- gsub("\n", "", text1[2])
    columns <- unlist(str_split(columns, pattern = " "))
    cols <- columns[columns != ""]
    text2 <- parser %>%
      html_nodes("td") %>% 
      html_text()
    
    draft_picks <- data.frame()
    for (i in seq(1, length(text2), 28))
    {
      row <- text2[i:(i + 27)]
      draft_picks <- rbind(draft_picks, row)
    }
    draft_picks$Rnd <- round
    draft_picks <- draft_picks %>% dplyr::select(Rnd, everything())
    colnames(draft_picks) <- cols
    draft_picks <- draft_picks[1:(length(draft_picks)-1)]
    draft_picks$Season <- year
    final_df <- rbind(final_df, draft_picks)
  }
  return(final_df)
}

# Create function to parse nfl pro football reference combine html website data
parse_nfl_combine <- function(start_year, end_year)
{
  final_df <- data.frame()
  for (year in start_year:end_year)
  {
    parser <- read_html(paste0("https://www.pro-football-reference.com/draft/",year,"-combine.htm"))
    columns <- parser %>%
      html_nodes("thead") %>% 
      html_text()
    columns <- gsub("\n", "", columns)
    columns <- unlist(str_split(columns, pattern = "   "))
    cols <- columns[columns != ""]
    
    player_names <- parser %>% 
      html_nodes("th.left") %>% 
      html_text() 
    player_names <- player_names[(player_names != "Pos" & player_names != "Player")]
    
    text <- parser %>% 
      html_nodes("td") %>% 
      html_text()
    
    combine <- data.frame()
    for (i in seq(1, length(text), 12))
    {
      row <- text[i:(i + 11)]
      combine <- rbind(combine, row)
    }
    combine$Player <- player_names     
    combine <- combine %>% dplyr::select(Player, everything())
    colnames(combine) <- cols
    combine$Season <- year
    final_df <- rbind(final_df, combine)
  }
  return(final_df)
}

# Create function to parse nfl pro football reference college stats html website data
parse_college_stats <- function(start_year, end_year)
{
  final_df <- data.frame()
  for (year in start_year:end_year)
  {
    parser <- read_html(paste0("https://www.sports-reference.com/cfb/years/",year,"-receiving.html"))

    columns <- parser %>%
      html_nodes("thead") %>% 
      html_text()
    columns <- gsub("\n", "", columns)
    columns <- unlist(str_split(columns, pattern = "   "))
    cols <- columns[columns != ""]
    cols <- cols[4:length(cols)]
    
    text <- parser %>% 
      html_nodes("td") %>% 
      html_text()
    
    college <- data.frame()
    for (i in seq(1, length(text), 16))
    {
      row <- text[i:(i + 15)]
      college <- rbind(college, row)
    }
    college$Rnk <- seq(1:nrow(college))    
    college <- college %>% dplyr::select(Rnk, everything())
    colnames(college) <- cols
    college$Season <- year
    final_df <- rbind(final_df, college)
  }
  return(final_df)
}

# Create function to parse nfl pro football reference nfl stats html website data
parse_nfl_receiving_stats <- function(start_year, end_year)
{
  final_df <- data.frame()
  for (year in start_year:end_year)
  {
    parser <- read_html(paste0("https://www.pro-football-reference.com/years/",year,"/receiving.htm"))
    columns <- parser %>%
      html_nodes("thead") %>% 
      html_text()
    columns <- gsub("\n", "", columns)
    columns <- unlist(str_split(columns, pattern = "   "))
    cols <- columns[columns != ""]
    
    text <- parser %>% 
      html_nodes("td") %>% 
      html_text()
    
    receiving <- data.frame()
    for (i in seq(1, length(text), 18))
    {
      row <- text[i:(i + 17)]
      receiving <- rbind(receiving, row)
    }
    receiving$Rk <- seq(1:nrow(receiving))    
    receiving <- receiving %>% dplyr::select(Rk, everything())
    colnames(receiving) <- cols
    receiving$Season <- year
    final_df <- rbind(final_df, receiving)
  }
  return(final_df)
}

# function to run k fold cross validation using x_vars to predict y_var for some 
# model type given (glmnet, rf, gbm, etc.) and make predictions on test set
run_model <- function(train, test, x_vars, y_var, model_type, tuneLength, k_fold, plot = FALSE)
{
  set.seed(2)
  control <- trainControl(number = k_fold)
  
  if (model_type == "glmnet")
  {
    model <- train(as.formula(paste0(y_var, "~ ", paste0(x_vars, collapse = " + "))), 
                   data = train, method = model_type, tuneLength = tuneLength, verbose = F,
                   metric = 'RMSE', trControl=control, preProcess = c('nzv','center','scale'))
  } else if (model_type == "rf")
  {
    model <- train(as.formula(paste0(y_var, "~ ", paste0(x_vars, collapse = " + "))), 
                   data = train, method = model_type, tuneLength = tuneLength, verbose = F,
                   metric = 'RMSE', trControl=control, ntree = 100)
  } else {
    model <- train(as.formula(paste0(y_var, "~ ", paste0(x_vars, collapse = " + "))), 
                   data = train, method = model_type, tuneLength = tuneLength, verbose = F,
                   metric = 'RMSE', trControl=control)
  }
  if ((model_type != 'gbm') & (plot == TRUE))
  {
    imp2 <- varImp(model)
    print(barchart(sort(rowMeans(imp2$importance), decreasing = T), 
                   main = paste0(model_type," Variable Importance Predicting ", y_var), 
                   xlab = "Average Level of Importance", ylab = "Variables"))
  }
  if (model_type == 'gbm')
  {
    var_imp <- summary(model)[2]
    labels <- row.names(var_imp)
    var_imp <- var_imp[1:20,]
    labels <- labels[1:20]
    df <- data.frame(labels, var_imp)
    save_plot <- ggplot(df, aes(x = reorder(labels, -var_imp), y = var_imp)) +
      geom_bar(stat = "identity", fill = "black") +
      ggtitle(paste0("GBM Variable Importance Predicting ", y_var)) + 
      coord_flip() + scale_y_continuous(name="Variable Important (0-100)") +
      scale_x_discrete(name="") +
      theme(plot.title=element_text(hjust=0.5,vjust=0,size=14,face = 'bold')) +
      theme(axis.text.x=element_text(vjust = .5, size=13,colour="#535353",face="bold")) +
      theme(axis.text.y=element_text(size=13,colour="#535353",face="bold")) +
      theme(axis.title.y=element_text(size=15,colour="#535353",face="bold",vjust=1.5)) +
      theme(axis.title.x=element_text(size=15,colour="#535353",face="bold",vjust=0))
  }
  preds <- predict(model, newdata = test)
  test$predictions <- preds
  if (model_type == "gbm")
  {
    return(list(model, preds, save_plot))
  } else {
    return(list(model, preds))
  }
}

# calculate mode, useful to use mode as aggregate function in group by statements
calc_mode <- function(x) {
  uniqx <- unique(na.omit(x))
  uniqx[which.max(tabulate(match(x, uniqx)))]
}

# function to create nice kable table output view of a dataframe
kable_table <- function(data, footnote = NULL, position = "center")
{
  tab <- kable(data, row.names = F) %>%
    kable_styling(bootstrap_options = c("striped", "hover", "condensed", 
                                        "responsive"), full_width = F, 
                  position = position, fixed_thead = T)
  if (!is.null(footnote))
  {
    tab <- tab %>% footnote(symbol = footnote)
  }
  return(tab)
}


################################################################################################
combine <- parse_nfl_combine(2000, 2020)
combine <- combine %>% filter(Pos == "WR") %>% dplyr::select(-College)
combine$Player <- trimws(combine$Player)
colnames(combine) <- c("Player","Pos","School","Ht","Wt","FortyYD","Vertical",
                       "Bench","Broad_Jump","Three_Cone","Shuttle","Drafted","Season_Drafted")
combine <- combine %>% 
  mutate(Ht = as.numeric(substr(Ht, start = 1, stop = 1)) * 12 + 
           as.numeric(substr(Ht, start = 3, stop = 3))) %>% 
  mutate_at(vars(Wt,FortyYD,Vertical,Bench,Broad_Jump,Three_Cone,Shuttle), as.numeric)




draft_picks <- parse_draft_class_future_nfl_stats(2000, 2020) 
draft_picks <- draft_picks %>% filter(Pos == "WR")
draft_picks$Player <- trimws(draft_picks$Player)
colnames(draft_picks) <- c("Rnd","Pick","Tm","Player","Pos","Age_Drafted","Most_Recent_NFL_Season",
                           "AP1","PB","St","CarAV","DrAV","Total_GP","Total_Pass_Cmp","Total_Pass_Att",
                           "Total_Pass_Yds","Total_Pass_TD","Total_Pass_Int","Total_Rush_Att","Total_Rush_Yds","Total_Rush_TD",
                           "Total_Num_Rec","Total_Rec_Yds","Total_Rec_TD","Total_Tackles","Total_Def_Int","Total_Def_Sk","College",
                           "Season_Drafted")
draft_picks <- draft_picks %>% 
  mutate_at(vars(Pick,Age_Drafted,Most_Recent_NFL_Season,AP1,PB,St,CarAV,DrAV,Total_GP,
                 Total_Pass_Cmp,Total_Pass_Att,Total_Pass_Yds,Total_Pass_TD,
                 Total_Pass_Int,Total_Rush_Att,Total_Rush_Yds,Total_Rush_TD,
                 Total_Num_Rec,Total_Rec_Yds,Total_Rec_TD,Total_Tackles,Total_Def_Int,
                 Total_Def_Sk), as.numeric)





college_stats <- parse_college_stats(2000, 2020)
college_stats$Player <- str_squish(str_replace_all(college_stats$Player, regex("[^ a-zA-Z'.]"), " "))
college_stats$Player <- trimws(college_stats$Player)
colnames(college_stats) <- c("Rk_C","Player","School","Conf","GP_C","Num_Rec_C","Rec_Yds_C",
                             "Rec_Yds_Per_Rec_C","Rec_TD_C","Rush_Att_C","Rush_Yds_C",
                             "Rush_Yds_Per_Att_C","Rush_TD_C","Num_Plays_C","Scrim_Yds_C",
                             "Scrim_Yds_Per_Play_C","Scrim_TD_C","College_Season")
college_stats <- college_stats %>% 
  mutate_at(vars(GP_C,Num_Rec_C,Rec_Yds_C,Rec_Yds_Per_Rec_C,Rec_TD_C,
                 Rush_Att_C,Rush_Yds_C,Rush_Yds_Per_Att_C,Rush_TD_C,
                 Num_Plays_C,Scrim_Yds_C,Scrim_Yds_Per_Play_C,Scrim_TD_C), as.numeric)





nfl_stats <- parse_nfl_receiving_stats(2000, 2020)
nfl_stats <- nfl_stats %>% filter(Pos == "WR")
nfl_stats$Player <- str_squish(str_replace_all(nfl_stats$Player, regex("[^ a-zA-Z'.]"), " "))
nfl_stats$Player <- trimws(nfl_stats$Player)
nfl_stats$`Ctch%` <- as.numeric(gsub("[\\%,]", "", nfl_stats$`Ctch%`))
colnames(nfl_stats) <- c("Rk","Player","Tm","Age","Pos","GP","GS","Num_Tgt","Num_Rec",
                         "Catch_Perc","Rec_Yds","Rec_Yds_Per_Rec","Rec_TD","First_Down_Rec",
                         "Longest_Rec","Rec_Yds_Per_Tgt","Num_Rec_Per_GP","Rec_Yds_Per_GP",
                         "Num_Fmb","NFL_Season")
nfl_stats <- nfl_stats %>% 
  mutate_at(vars(Age,GP,GS,Num_Tgt,Num_Rec,Rec_Yds,Rec_Yds_Per_Rec,Rec_TD,
                 First_Down_Rec,Longest_Rec,Rec_Yds_Per_Tgt,Num_Rec_Per_GP,
                 Rec_Yds_Per_GP,Num_Fmb), as.numeric)

###########################################################################################
# Create college game, play, and reception level statistics for each player adding across all years
college_stats2 <- college_stats %>% 
  group_by(Player, Conf) %>% 
  summarise(Num_College_Seasons = n_distinct(College_Season),
            Conf = calc_mode(Conf),
            Num_Rec_Per_Game_C = sum(Num_Rec_C)  / sum(GP_C),
            Rec_Yds_Per_Game_C = sum(Rec_Yds_C) / sum(GP_C),
            Rec_Yds_Per_Rec_C = sum(Rec_Yds_C) / sum(Num_Rec_C),
            Rec_TD_Per_Game_C = sum(Rec_TD_C) / sum(GP_C),
            Num_Plays_Per_Game_C = sum(Num_Plays_C) / sum(GP_C),
            Scrim_Yds_Per_Play_C = sum(Scrim_Yds_C) / sum(Num_Plays_C),
            Scrim_TD_Per_Game_C = sum(Scrim_TD_C) / sum(GP_C),
            GP_C = sum(GP_C), 
            Last_Season_College = max(College_Season), 
            Season_Drafted = Last_Season_College + 1, 
            Total_Rec_Yds_C = sum(Rec_Yds_C),
            Total_TD_C = sum(Rec_TD_C),
            Weights_C = sum(Num_Rec_C), .groups = "drop",) %>% 
  arrange(-Num_Rec_Per_Game_C)

# Create nfl game, play, and reception level metrics for each player adding across all first 3 years or 
# all years if player has less than 3 years experience
nfl_stats2 <- nfl_stats %>% 
  group_by(Player, Pos) %>% 
  mutate(Rookie_Season = min(NFL_Season)) %>% 
  filter(NFL_Season - Rookie_Season < 3) %>% 
  group_by(Player, Pos) %>% 
  summarize(Num_Tgt_Per_Game = sum(Num_Tgt) / sum(GP),
            Avg_Catch_Perc = round(sum(Num_Rec) / sum(Num_Tgt),3) * 100,
            Num_Rec_Per_Game = sum(Num_Rec) / sum(GP),
            Rec_Yds_Per_Game = sum(Rec_Yds) / sum(GP),
            Rec_Yds_Per_Rec = sum(Rec_Yds) / sum(Num_Rec),
            Rec_TD_Per_Game = sum(Rec_TD) / sum(GP),
            First_Down_Rec_Per_Game = sum(First_Down_Rec) / sum(GP), 
            Rec_Yds_Per_Tgt = sum(Rec_Yds) / sum(Num_Tgt), 
            Weights = sum(Num_Tgt), .groups = "drop")

# combine all data together to create one large dataset to build models from
df <- college_stats2 %>%
  mutate(Player = str_trim(str_remove(Player, "Jr."))) %>%
  inner_join(combine, by = c("Player","Season_Drafted")) %>% 
  left_join(nfl_stats2 %>% dplyr::select(-Pos), by = c("Player")) %>% 
  left_join(draft_picks[, c("Age_Drafted","Season_Drafted","Player")],
             by = c("Player","Season_Drafted")) %>% 
  mutate(Conf = case_when(Conf %in% c("Pac-10","Pac-12") ~ "Pac-12", TRUE ~ Conf),
         Power_Conf = case_when(Conf %in% c("ACC","Big Ten","Big 12","Pac-12","SEC") ~ 1, TRUE ~ 0),
         Conf = as.factor(Conf),
         Power_Conf = as.factor(Power_Conf),
         Is_Alabama = case_when(School == "Alabama" ~ 1, TRUE ~ 0),
         Is_Alabama = as.factor(Is_Alabama),
         Ht = case_when(Ht == 61 ~ 71, TRUE ~ Ht),
         Ht_m = Ht / 39.37,
         Wt_kg = Wt / 2.205,
         BMI = Wt_kg/ (Ht_m^2))

# Split data into train, test splits
train <- df %>% filter(Season_Drafted < 2020)
test <- df %>% filter(Season_Drafted == 2020)


# Predictor Variables to use
x_vars <- c("Conf","Power_Conf","Num_College_Seasons","FortyYD","Vertical",
            "Num_Rec_Per_Game_C","Rec_Yds_Per_Game_C","Age_Drafted",
            "Rec_Yds_Per_Rec_C","Rec_TD_Per_Game_C","Num_Plays_Per_Game_C",
            "Scrim_Yds_Per_Play_C","Scrim_TD_Per_Game_C","Ht","Wt","Broad_Jump",
            "Three_Cone","Shuttle","BMI","Total_Rec_Yds_C","Total_TD_C")

#####################################################################################################
y_var <- "Num_Tgt_Per_Game"
vars <- c("Player",y_var,x_vars)
train2 <- train[, vars]

train_pre_obj <- preProcess(as.data.frame(train2[, c(x_vars, y_var)]),method = "bagImpute", k = 5)
training <- predict(train_pre_obj, train2[, c(x_vars, y_var)])

test2 <- test[, c("Player",x_vars)]
test_pre_obj <- preProcess(as.data.frame(test2[, x_vars]), method = "bagImpute", k = 5)
testing <- predict(test_pre_obj, test2[, x_vars])
testing$Player <- test2$Player

model_tgt_per_game <- run_model(train = training, test = testing, y_var = y_var, x_vars = x_vars, 
               model_type = "glmnet", k_fold = 5, tuneLength = 5, plot = T)

model_tgt_per_game[[1]]

coefs <- coef(model_tgt_per_game[[1]]$finalModel, model_tgt_per_game[[1]]$bestTune$lambda)
variables <- rownames(coefs)
estimates <- round(coefs[,1],5)
results <- data.frame(Variables = variables, Estimates = estimates)
kable_table(results, "Coefficients Predicting Number of Targets Per Game")
##############################################################################################
y_var <- "Avg_Catch_Perc"
vars <- c("Player",y_var, x_vars)
train2 <- train[, vars]

train_pre_obj <- preProcess(as.data.frame(train2[, c(x_vars, y_var)]),method = "bagImpute", k = 5)
training <- predict(train_pre_obj, train2[, c(x_vars, y_var)])

test2 <- test[, c("Player",x_vars)]
test_pre_obj <- preProcess(as.data.frame(test2[, x_vars]), method = "bagImpute", k = 5)
testing <- predict(test_pre_obj, test2[, x_vars])
testing$Player <- test2$Player

model_avg_catch_perc <- run_model(train = training, test = testing, y_var = y_var, x_vars = x_vars, 
                                model_type = "glmnet", k_fold = 5, tuneLength = 5, plot = T)

model_avg_catch_perc[[1]]

coefs <- coef(model_avg_catch_perc[[1]]$finalModel, model_avg_catch_perc[[1]]$bestTune$lambda)
variables <- rownames(coefs)
estimates <- round(coefs[,1],5)
results <- data.frame(Variables = variables, Estimates = estimates)
kable_table(results, "Coefficients Predicting Avg. Catch %")
#############################################################################################
y_var <- "Num_Rec_Per_Game"
vars <- c("Player",y_var,x_vars)
train2 <- train[, vars]

train_pre_obj <- preProcess(as.data.frame(train2[, c(x_vars, y_var)]),method = "bagImpute", k = 5)
training <- predict(train_pre_obj, train2[, c(x_vars, y_var)])

test2 <- test[, c("Player",x_vars)]
test_pre_obj <- preProcess(as.data.frame(test2[, x_vars]), method = "bagImpute", k = 5)
testing <- predict(test_pre_obj, test2[, x_vars])
testing$Player <- test2$Player

model_num_rec_per_game <- run_model(train = training, test = testing, y_var = y_var, x_vars = x_vars, 
                                model_type = "glmnet", k_fold = 5, tuneLength = 5, plot = T)

model_num_rec_per_game[[1]]

coefs <- coef(model_num_rec_per_game[[1]]$finalModel, model_num_rec_per_game[[1]]$bestTune$lambda)
variables <- rownames(coefs)
estimates <- round(coefs[,1],5)
results <- data.frame(Variables = variables, Estimates = estimates)
kable_table(results, "Coefficients Predicting Number of Receptions Per Game")
############################################################################################
y_var <- "Rec_Yds_Per_Game"
vars <- c("Player",y_var,x_vars)
train2 <- train[, vars]

train_pre_obj <- preProcess(as.data.frame(train2[, c(x_vars, y_var)]),method = "bagImpute", k = 5)
training <- predict(train_pre_obj, train2[, c(x_vars, y_var)])

test2 <- test[, c("Player",x_vars)]
test_pre_obj <- preProcess(as.data.frame(test2[, x_vars]), method = "bagImpute", k = 5)
testing <- predict(test_pre_obj, test2[, x_vars])
testing$Player <- test2$Player

model_rec_yds_per_game <- run_model(train = training, test = testing, y_var = y_var, x_vars = x_vars, 
                                    model_type = "glmnet", k_fold = 5, tuneLength = 5, plot = T)

model_rec_yds_per_game[[1]]

coefs <- coef(model_rec_yds_per_game[[1]]$finalModel, model_rec_yds_per_game[[1]]$bestTune$lambda)
variables <- rownames(coefs)
estimates <- round(coefs[,1],5)
results <- data.frame(Variables = variables, Estimates = estimates)
kable_table(results, "Coefficients Predicting Receiving Yds Per Game")
##############################################################################################
y_var <- "Rec_Yds_Per_Rec"
vars <- c("Player",y_var,x_vars)
train2 <- train[, vars]

train_pre_obj <- preProcess(as.data.frame(train2[, c(x_vars, y_var)]),method = "bagImpute", k = 5)
training <- predict(train_pre_obj, train2[, c(x_vars, y_var)])

test2 <- test[, c("Player",x_vars)]
test_pre_obj <- preProcess(as.data.frame(test2[, x_vars]), method = "bagImpute", k = 5)
testing <- predict(test_pre_obj, test2[, x_vars])
testing$Player <- test2$Player

model_rec_yds_per_rec <- run_model(train = training, test = testing, y_var = y_var, x_vars = x_vars, 
                                    model_type = "glmnet", k_fold = 5, tuneLength = 5, plot = T)

model_rec_yds_per_rec[[1]]

coefs <- coef(model_rec_yds_per_rec[[1]]$finalModel, model_rec_yds_per_rec[[1]]$bestTune$lambda)
variables <- rownames(coefs)
estimates <- round(coefs[,1],5)
results <- data.frame(Variables = variables, Estimates = estimates)
kable_table(results, "Coefficients Predicting Receiving Yds Per Reception")
################################################################################################
y_var <- "Rec_TD_Per_Game"
vars <- c("Player",y_var,x_vars)
train2 <- train[, vars]

train_pre_obj <- preProcess(as.data.frame(train2[, c(x_vars, y_var)]),method = "bagImpute", k = 5)
training <- predict(train_pre_obj, train2[, c(x_vars, y_var)])

test2 <- test[, c("Player",x_vars)]
test_pre_obj <- preProcess(as.data.frame(test2[, x_vars]), method = "bagImpute", k = 5)
testing <- predict(test_pre_obj, test2[, x_vars])
testing$Player <- test2$Player

model_rec_td_per_game <- run_model(train = training, test = testing, y_var = y_var, x_vars = x_vars, 
                                   model_type = "glmnet", k_fold = 5, tuneLength = 5, plot = T)

model_rec_td_per_game[[1]]

coefs <- coef(model_rec_td_per_game[[1]]$finalModel, model_rec_td_per_game[[1]]$bestTune$lambda)
variables <- rownames(coefs)
estimates <- round(coefs[,1],5)
results <- data.frame(Variables = variables, Estimates = estimates)
kable_table(results, "Coefficients Predicting Receiving TDs Per Game")
################################################################################################
y_var <- "First_Down_Rec_Per_Game"
vars <- c("Player",y_var,x_vars)
train2 <- train[, vars]

train_pre_obj <- preProcess(as.data.frame(train2[, c(x_vars, y_var)]),method = "bagImpute", k = 5)
training <- predict(train_pre_obj, train2[, c(x_vars, y_var)])

test2 <- test[, c("Player",x_vars)]
test_pre_obj <- preProcess(as.data.frame(test2[, x_vars]), method = "bagImpute", k = 5)
testing <- predict(test_pre_obj, test2[, x_vars])
testing$Player <- test2$Player

model_first_down_rec_per_game <- run_model(train = training, test = testing, y_var = y_var, x_vars = x_vars, 
                                   model_type = "glmnet", k_fold = 5, tuneLength = 5, plot = T)

model_first_down_rec_per_game[[1]]

coefs <- coef(model_first_down_rec_per_game[[1]]$finalModel, model_first_down_rec_per_game[[1]]$bestTune$lambda)
variables <- rownames(coefs)
estimates <- round(coefs[,1],5)
results <- data.frame(Variables = variables, Estimates = estimates)
kable_table(results, "Coefficients Predicting 1st Down Receptions Per Game")
################################################################################################
y_var <- "Rec_Yds_Per_Tgt"
vars <- c("Player",y_var,x_vars)
train2 <- train[, vars]

train_pre_obj <- preProcess(as.data.frame(train2[, c(x_vars, y_var)]),method = "bagImpute", k = 5)
training <- predict(train_pre_obj, train2[, c(x_vars, y_var)])

test2 <- test[, c("Player",x_vars)]
test_pre_obj <- preProcess(as.data.frame(test2[, x_vars]), method = "bagImpute", k = 5)
testing <- predict(test_pre_obj, test2[, x_vars])
testing$Player <- test2$Player

model_rec_yds_per_tgt <- run_model(train = training, test = testing, y_var = y_var, x_vars = x_vars, 
                                           model_type = "glmnet", k_fold = 5, tuneLength = 5, plot = T)

model_rec_yds_per_tgt[[1]]

coefs <- coef(model_rec_yds_per_tgt[[1]]$finalModel, model_rec_yds_per_tgt[[1]]$bestTune$lambda)
variables <- rownames(coefs)
estimates <- round(coefs[,1],5)
results <- data.frame(Variables = variables, Estimates = estimates)
kable_table(results, "Coefficients Predicting Receiving Yds Per Target")
##############################################################################################
testing$Tgt_Per_Game_Pred <- model_tgt_per_game[[2]]
testing$Avg_Catch_Perc_Pred <- model_avg_catch_perc[[2]]
testing$Num_Rec_Per_Game_Pred <- model_num_rec_per_game[[2]]
testing$Rec_Yds_Per_Game_Pred <- model_rec_yds_per_game[[2]]
testing$Rec_Yds_Per_Rec_Pred <- model_rec_yds_per_rec[[2]]
testing$Rec_TD_Per_Game_Pred <- model_rec_td_per_game[[2]]
testing$First_Down_Rec_Per_Game_Pred <- model_first_down_rec_per_game[[2]]
testing$Rec_Yds_Per_Tgt_Pred <- model_rec_yds_per_tgt[[2]]
#############################################################################################
testing2 <- testing %>% 
  select(Tgt_Per_Game_Pred, Num_Rec_Per_Game_Pred, Rec_Yds_Per_Game_Pred, Avg_Catch_Perc_Pred,
         Rec_Yds_Per_Rec_Pred, Rec_TD_Per_Game_Pred, First_Down_Rec_Per_Game_Pred, 
         Rec_Yds_Per_Tgt_Pred) %>%
  mutate(Tgt_Per_Game_Pred = -Tgt_Per_Game_Pred, 
         Avg_Catch_Perc_Pred = -Avg_Catch_Perc_Pred,
         Num_Rec_Per_Game_Pred = -Num_Rec_Per_Game_Pred,
         Rec_Yds_Per_Game_Pred = -Rec_Yds_Per_Game_Pred,
         Rec_Yds_Per_Rec_Pred = -Rec_Yds_Per_Rec_Pred,
         Rec_TD_Per_Game_Pred = -Rec_TD_Per_Game_Pred, 
         First_Down_Rec_Per_Game_Pred = -First_Down_Rec_Per_Game_Pred,
         Rec_Yds_Per_Tgt_Pred = -Rec_Yds_Per_Tgt_Pred) %>% 
  apply(., 2, rank) %>% as.data.frame()
testing2 <- testing2 %>%
  rowwise() %>%
  mutate(MedianRank = median(c_across(where(is.numeric)), na.rm = TRUE))
testing2$Player <- testing$Player
testing2 <- testing2 %>% inner_join(test[, c("Player","Drafted")], by = "Player") %>% 
  arrange(MedianRank) %>%
  select(Player, MedianRank, Tgt_Per_Game_Pred, Avg_Catch_Perc_Pred, Num_Rec_Per_Game_Pred,
         Rec_Yds_Per_Game_Pred, Rec_Yds_Per_Rec_Pred, Rec_TD_Per_Game_Pred, 
         First_Down_Rec_Per_Game_Pred, Rec_Yds_Per_Tgt_Pred, Drafted) %>% 
  rename("Tgt Per G" = "Tgt_Per_Game_Pred", "Avg Catch %" = "Avg_Catch_Perc_Pred",
         "Num Rec Per G" = "Num_Rec_Per_Game_Pred","Rec Yds Per G" = "Rec_Yds_Per_Game_Pred",
         "Rec Yds Per Rec" = "Rec_Yds_Per_Rec_Pred", "Rec TD Per G" = "Rec_TD_Per_Game_Pred", 
         "1st Down Rec Per G" = "First_Down_Rec_Per_Game_Pred", "Rec Yds Per Tgt" = "Rec_Yds_Per_Tgt_Pred")


actual <- draft_picks %>% 
  filter(Season_Drafted == 2020) %>% 
  mutate(Num_Rec_Per_Game = Total_Num_Rec / Total_GP,
         Rec_Yds_Per_Game = Total_Rec_Yds / Total_GP,
         Rec_Yds_Per_Rec = Total_Rec_Yds / Total_Num_Rec,
         Rec_TD_Per_Game = Total_Rec_TD / Total_GP) %>% 
  mutate_if(is.numeric, round, 2) %>% 
  select(Rnd, Pick, Player, Tm, Age_Drafted, Num_Rec_Per_Game, 
         Rec_Yds_Per_Game, Rec_Yds_Per_Rec, Rec_TD_Per_Game) %>%
  arrange(Rec_Yds_Per_Game)
  


df %>% filter(Season_Drafted == 2020) %>% View()
