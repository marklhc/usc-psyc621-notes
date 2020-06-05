library(ggplot2)

mmp_brm <- function(object, x = NULL, prob = 0.95, size = 0.8, 
                    plot_pi = FALSE, jitter = FALSE, 
                    smooth_method = "auto") {
  dat <- object$data
  post_mu <- fitted(object, scale = "response")
  colnames(post_mu) <- c("mu", "mu_se", "lwr_ci", "upr_ci")
  df_plot <- cbind(dat, post_mu)
  if (is.null(x)) {
    lin_pred <- fitted(object, scale = "linear")
    df_plot$lin_pred <- lin_pred[ , 1]
    x <- "lin_pred"
  }
  x_sd <- sd(df_plot[[x]])
  p <- ggplot(aes_string(x = paste0("`", x, "`"), 
                         y = paste0("`", names(dat)[1], "`")), data = df_plot) + 
    # Add a layer of predictive intervals
    geom_ribbon(aes(ymin = predict(loess(as.formula(paste("lwr_ci ~", x)), 
                                         data = df_plot)), 
                    ymax = predict(loess(as.formula(paste("upr_ci ~", x)), 
                                         data = df_plot))), 
                fill = "skyblue", alpha = 0.3) + 
    geom_smooth(aes(y = mu, col = "Model"), se = FALSE, 
                method = smooth_method) + 
    geom_smooth(aes(col = "Data"), se = FALSE, linetype = "dashed", 
                method = smooth_method) + 
    theme(legend.position = "bottom") + 
    scale_color_manual(values = c("red", "blue"), name = "")
  if (jitter) {
    p <- p + geom_jitter(size = size, width = x_sd * .1, height = .02)
  } else {
    p <- p + geom_point(size = size)
  }
  if (plot_pi) {
    pi <- predictive_interval(object, prob = prob)
    colnames(pi) <- c("lwr_pi", "upr_pi")  # change the names for convenienc
    # Combine the PIs with the original data
    df_plot <- cbind(df_plot, pi)
    p <- p + geom_smooth(aes(y = upr_pi), data = df_plot, linetype = "longdash", 
                         se = FALSE, size = 0.5, col = "green", 
                         method = smooth_method) + 
      geom_smooth(aes(y = lwr_pi), data = df_plot, linetype = "longdash", 
                  se = FALSE, size = 0.5, col = "green", 
                  method = smooth_method)
  }
  p
}