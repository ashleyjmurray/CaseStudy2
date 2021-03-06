library(dplyr)

final <- read.csv("final.csv")

final <- final %>%
  mutate(S2 = as.factor(case_when(subject == 'S2' ~ 1,
                                  TRUE ~ 0)),
         S3 = as.factor(case_when(subject == 'S3' ~ 1,
                                  TRUE ~ 0)),
         S4 = as.factor(case_when(subject == 'S4' ~ 1,
                                  TRUE ~ 0)),
         S5 = as.factor(case_when(subject == 'S5' ~ 1,
                                  TRUE ~ 0)),
         S6 = as.factor(case_when(subject == 'S6' ~ 1,
                                  TRUE ~ 0)),
         S7 = as.factor(case_when(subject == 'S7' ~ 1,
                                  TRUE ~ 0)),
         S8 = as.factor(case_when(subject == 'S8' ~ 1,
                                  TRUE ~ 0)),
         S9 = as.factor(case_when(subject == 'S9' ~ 1,
                                  TRUE ~ 0)),
         S10 = as.factor(case_when(subject == 'S10' ~ 1,
                                   TRUE ~ 0)),
         S11 = as.factor(case_when(subject == 'S11' ~ 1,
                                   TRUE ~ 0)),
         S13 = as.factor(case_when(subject == 'S13' ~ 1,
                                   TRUE ~ 0)),
         S14 = as.factor(case_when(subject == 'S14' ~ 1,
                                   TRUE ~ 0)),
         S15 = as.factor(case_when(subject == 'S15' ~ 1,
                                   TRUE ~ 0)),
         S16 = as.factor(case_when(subject == 'S16' ~ 1,
                                   TRUE ~ 0)),
         S17 = as.factor(case_when(subject == 'S17' ~ 1,
                                   TRUE ~ 0))
  )

final <- final %>%
  select(-subject)

final$label <- as.factor(final$label)

final <- within(final, label <- as.factor(label))
final <- within(final, label <- relevel(label, ref = 2))


final_model <- glm(label ~ S2 + S3 + S4 + S5 + S6 + S7 + S11 + S13 + S14 + S15 + S16 +
                     HRV_MeanNN +
                     HRV_MeanNN*S2 +
                     HRV_MeanNN*S3 +
                     HRV_MeanNN*S4+
                     HRV_MeanNN*S5+
                     HRV_MeanNN*S6+
                     HRV_MeanNN*S7+
                     HRV_MeanNN*S8+
                     HRV_MeanNN*S9+
                     HRV_MeanNN*S10+
                     HRV_MeanNN*S11+
                     HRV_MeanNN*S13+
                     HRV_MeanNN*S14+
                     HRV_MeanNN*S15+
                     HRV_MeanNN*S16+
                     
                     eda_slope_wr+
                     eda_slope_wr*S2+
                     eda_slope_wr*S3+
                     eda_slope_wr*S4+
                     eda_slope_wr*S5+
                     eda_slope_wr*S6+
                     eda_slope_wr*S7+
                     eda_slope_wr*S8+
                     eda_slope_wr*S9+
                     eda_slope_wr*S10+
                     eda_slope_wr*S11+
                     eda_slope_wr*S13+
                     eda_slope_wr*S14+
                     eda_slope_wr*S15+
                     eda_slope_wr*S16+
                     
                     bvp_nintieth_quantile+
                     bvp_nintieth_quantile*S2+
                     bvp_nintieth_quantile*S3+
                     bvp_nintieth_quantile*S4+
                     bvp_nintieth_quantile*S5+
                     bvp_nintieth_quantile*S6+
                     bvp_nintieth_quantile*S7+
                     bvp_nintieth_quantile*S8+
                     bvp_nintieth_quantile*S9+
                     bvp_nintieth_quantile*S10+
                     bvp_nintieth_quantile*S11+
                     bvp_nintieth_quantile*S13+
                     bvp_nintieth_quantile*S14+
                     bvp_nintieth_quantile*S15+
                     bvp_nintieth_quantile*S16+
                     
                     acc_x_mean+
                     acc_x_mean*S2+
                     acc_x_mean*S3+
                     acc_x_mean*S4+
                     acc_x_mean*S5+
                     acc_x_mean*S6+
                     acc_x_mean*S7+
                     acc_x_mean*S8+
                     acc_x_mean*S9+
                     acc_x_mean*S10+
                     acc_x_mean*S11+
                     acc_x_mean*S13+
                     acc_x_mean*S14+
                     acc_x_mean*S15+
                     acc_x_mean*S16+
                     
                     temp_wr_standard_deviation+
                     temp_wr_standard_deviation*S2+
                     temp_wr_standard_deviation*S3+
                     temp_wr_standard_deviation*S4+
                     temp_wr_standard_deviation*S5+
                     temp_wr_standard_deviation*S6+
                     temp_wr_standard_deviation*S7+
                     temp_wr_standard_deviation*S8+
                     temp_wr_standard_deviation*S9+
                     temp_wr_standard_deviation*S10+
                     temp_wr_standard_deviation*S11+
                     temp_wr_standard_deviation*S13+
                     temp_wr_standard_deviation*S14+
                     temp_wr_standard_deviation*S15+
                     temp_wr_standard_deviation*S16+
                     
                     ie_ratio+
                     ie_ratio*S2+
                     ie_ratio*S3+
                     ie_ratio*S4+
                     ie_ratio*S5+
                     ie_ratio*S6+
                     ie_ratio*S7+
                     ie_ratio*S8+
                     ie_ratio*S9+
                     ie_ratio*S10+
                     ie_ratio*S11+
                     ie_ratio*S13+
                     ie_ratio*S14+
                     ie_ratio*S15+
                     ie_ratio*S16, family = binomial, data = final)

summary(final_model)


