
# To be added after Siow Meng's code (after the big "for" loop)

library(reshape)
library(ggplot2)
library(ggthemes)

###

vPerf <- melt(validationPerf, id.vars = "weight")

ggplot(vPerf) +
    geom_line(aes(x = weight, y = value, color = variable)) +
    geom_hline(yintercept = .25, linetype = 3, alpha = .5) +
    geom_hline(yintercept =  .4, linetype = 3, alpha = .5) +
    geom_hline(yintercept =  .5, linetype = 3, alpha = .5) +
    theme_tufte() +
    scale_color_manual(values = c("purple", "#E69F00", "#56B4E9", "black"), 
                       name = "Performance\nMeasures",
                       breaks = c("sens", "prec", "spec", "acc"),
                       labels = c("Sensitivity", "Precision", "Specificity", "Accuracy")) +
    labs(x = "False Positive's weight", y = "Percentage (%)") +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, .2), labels = seq(0, 1, .2) * 100)

###

