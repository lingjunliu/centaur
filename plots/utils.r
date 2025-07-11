stat_tests <- function(data1, data2, suffix) {
  # Statistical test (KS test)
  
  ksPvalueSota <- ks.test(data1, "pnorm", exact = FALSE)
  # D = 1, p-value < 2.2e-16 | not normal < 0.05 < normal
  
#   install.packages("nortest", dependencies = TRUE)
  library(nortest)
  adPvalueSota <- ad.test(data1)
  # AD test A = 0.82877, p-value = 0.02846 | not normal < 0.05 < normal
  
  ksPvalueTname <- ks.test(data2, "pnorm", exact = FALSE)
  # D = 1, p-value < 2.2e-16 | not normal < 0.05 < normal
  
  adPvalueTname <- ad.test(data2)
  # AD test A = 1.002, p-value = 0.01036 | not normal < 0.05 < normal
  
  ksPvalueTwoSample <- ks.test(data1, data2)
  # D = 0.2069, p-value = 0.5554 | different distribution < 0.05 < same distribution
  # Null hypothesis can not be rejected -> data comes from same distribution
  
  wilcoxonPvalue <- wilcox.test(data1, data2, exact = FALSE)
  # W = 368, p-value = 0.4187 | significant < 0.05 < insignificant
  # Not statistically significant
  
  tTestPvalue <- t.test(data1, data2)
  
#   install.packages("lsr", dependencies = TRUE)
  library(lsr)
  cohensDvalue <- cohensD(data1, data2)
  n_apis <- length(data1)
  options(scipen=0)
  cat("\n")
  cat(sprintf("\\newcommand{\\nAPIs%s}{%s} %% | # of APIs\n", suffix, n_apis))
  cat(sprintf("\\newcommand{\\ksPvalueSota%s}{%.3g} %% %s | not normal < 0.05 < normal\n", suffix, ksPvalueSota$p.value, ksPvalueSota$p.value))
  cat(sprintf("\\newcommand{\\ksPvalueTname%s}{%.3g} %% %s | not normal < 0.05 < normal\n", suffix, ksPvalueTname$p.value, ksPvalueTname$p.value))
  cat(sprintf("\\newcommand{\\adPvalueSota%s}{%.3g} %% %s | not normal < 0.05 < normal\n", suffix, adPvalueSota$p.value, adPvalueSota$p.value))
  cat(sprintf("\\newcommand{\\adPvalueTname%s}{%.3g} %% %s | not normal < 0.05 < normal\n", suffix, adPvalueTname$p.value, adPvalueTname$p.value))
  
  cat(sprintf("\\newcommand{\\ksPvalueTwoSample%s}{%.3g} %% %s | different distribution < 0.05 < same distribution\n", suffix, ksPvalueTwoSample$p.value, ksPvalueTwoSample$p.value))
  cat(sprintf("\\newcommand{\\wilcoxonPvalue%s}{%.3g} %% %s | significant < 0.05 < insignificant\n", suffix, wilcoxonPvalue$p.value, wilcoxonPvalue$p.value))
  cat(sprintf("\\newcommand{\\tTestPvalue%s}{%.3g} %% %s | significant < 0.05 < insignificant\n", suffix, tTestPvalue$p.value, tTestPvalue$p.value))
  
  cat(sprintf("\\newcommand{\\cohensDvalue%s}{%.2f} %% %s | small: 0.2 | medium: 0.5 | large: 0.8\n", suffix, cohensDvalue, cohensDvalue))
  cat("\n%---------------------------------\n")
}



# Histogram and Kurtosis

# Freefuzz <- freefuzz
# hist(Freefuzz, xlab = sprintf("Freefuzz ran with 300 inputs on %.3g apis", length(data$api)))
# Graybox <- graybox
# hist(Graybox, xlab = sprintf("Graybox ran with 300 inputs on %.3g apis", length(data$api)))

# library(moments)
# kurtosisSota <- kurtosis(freefuzz)
# kurtosisTname <- kurtosis(graybox)
# 
# print(sprintf("kurtosisSota: %s | platykurtic < 3 < leptokurtic", kurtosisSota))
# print(sprintf("kurtosisTname: %s | platykurtic < 3 < leptokurtic", kurtosisTname))