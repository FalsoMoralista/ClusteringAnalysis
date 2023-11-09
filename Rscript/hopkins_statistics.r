#install.packages("tidyverse", repos = "http://cran.us.r-project.org")
#install.packages("conflicted", repos = "http://cran.us.r-project.org")

library(conflicted)
library(reticulate)
library(hopkins)
library(readr)
library(tidyverse)

#library(rTorch)
#library(torch)

features <- read_csv('../features/ImageNET/inet_features.csv',show_col_types = FALSE)
features <- subset(features, select = -img_path) # Drop column

filter <- dplyr::filter

head(features, 10)


averages <- list()
sds <- list()
id_list <- list()
class_ids <- list.dirs(path = "../../../train", full.names = FALSE)
for(id in class_ids[-1]){
  id = strtoi(id, base = 0L)

  df_subset <- features %>% filter( 
    class_id == id # 10000918 or 10003327 (tests)
  ) 
  # Remove id columns.
  df_subset <- subset(df_subset, select = -class_id)
  df_subset <- df_subset[, -which(names(df_subset) == "...1")]
  #print('Executing...')
  #print(df_subset)
  results <- list() 

  if(nrow(df_subset) >= 3){
    m = as.integer((1/3) * nrow(df_subset))
    
    for(i in 1:100){
      results <- append(results, hopkins(df_subset, m=m, d=1 ,method="simple"))
    }
    #cat(sprintf("Results for class: %d", id))
    #print("Mean:")
    m <- mean(unlist(results)) # Compute the average Hopkins statistics
    #print(m)
    averages <- append(averages, m) # Append the averages into a list
    #print("Std:")
    std <- sd(unlist(results)) # Repeat for the standard deviation
    #print(std)
    sds <- append(sds, std) # Append
    id_list <- append(id_list, id) # Record the respective class ID.
  }
}

entries <- list(id=id_list, avg=averages, std=sds)
data <-  as.data.frame(do.call(cbind, entries))

data <- apply(data, 2, as.character)  
print(data) 
write.csv(data, "output2.csv", row.names=FALSE, quote=FALSE)



#my_data <- py_load_object("../features/ImageNET/features.pkl")



#r_df = dict_to_df_gpt3(my_data)
#head(r_df)
#extracted_features <- py_to_r(my_data)
#head(extracted_features)
