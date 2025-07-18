#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly=TRUE)

sota <- "Titanfuzz"
if (length(args) > 0) {
  sota <- args[1]
}

tname <- "SLATE"
tname_alias <- "Centaur"

mode <- "cov"   # "cov" or "val"
if (length(args) > 1) {
  mode <- args[2]
}

if (length(args) > 2) {
  csv <- args[3]
}

if (length(args) > 3) {
  lib <- args[4]
} else {
  lib <- "torch"
}

if (mode == "cov") {
  ylabel <- "# branches covered"
} else if (mode == "val") {
  ylabel <- "Validity Ratio (%)"
} else {
  ylabel <- sprintf("Comparison - %s", mode)
}
font_size <- 1.2
create_pdf <- TRUE

source("utils.r")

if (sota == "FreeFuzz") {
  sota_color <- "royalblue2"
} else if (sota == "DeepREL") {
  sota_color <- "orange"
} else if (sota == "Titanfuzz") {
  sota_color <- "pink"
} else if (sota == "ACETest") {
  sota_color <- "purple"
} else if (sota == "Pathfinder") {
  sota_color <- "darkcyan"
}

if (sota == "FreeFuzz") {
  suffix <- "FF"
} else if (sota == "DeepREL") {
  suffix <- "DR"
} else if (sota == "Titanfuzz") {
  suffix <- "TF"
} else if (sota == "ACETest") {
  suffix <- "AC"
} else if (sota == "Pathfinder") {
  suffix <- "PF"
}

if (lib == "tf") {
  suffix <- paste0(suffix, "tf")
}

if (mode == "val") {
  suffix <- paste0(suffix, "Val")
}

if (sota == "Pathfinder") {
  if (lib == "torch") {
    ylim <- c(2500, 11500)
  } else {
    ylim <- c(1000, 8500)
  }
} else if (lib == "torch") {
  ylim <- c(9000, 11500)
} else {
  ylim <- c(2000, 9000)
}

if (mode == "val") {
  ylim <- c(0, 100)
}

data <- read.csv(csv)
if (lib == "torch") {
  filename <- sprintf("data/%s_vs_%s_%s.pdf", sota, tname, mode)
} else {
  filename <- sprintf("data/%s_vs_%s_%s_tf.pdf", sota, tname, mode)
}

sota_col <- data[[sota]]
tname_col <- data[[tname]]

if (create_pdf) {
  pdf(file = filename, width = 6, height = 8)
}

# Box plot
par(cex.axis = font_size)
par(cex.lab = font_size)
box_plot <- boxplot(sota_col, tname_col,
                    names = c(sota, tname_alias),
                    col = c(sota_color, "darkgreen"),
                    ylab = ylabel,
                    notch = FALSE,
                    ylim = ylim,
                    boxwex = .5)
if (create_pdf) {
  dev.off()
}

sota_mean <- mean(sota_col)
tname_mean <- mean(tname_col)
diff <- tname_mean - sota_mean

if (diff > 0) {
  comparison <- "better"
} else {
  comparison <- "worse"
}

cat("%---------------------------------")
cat(sprintf("\n%% %s is %s than %s on average (Diff: %.2f)\n", tname, comparison, sota, diff))
cat("%---------------------------------")
cat(sprintf("\n%% %s\n", sota))
stat_tests(sota_col, tname_col, suffix)