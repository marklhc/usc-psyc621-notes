# From https://github.com/leifeld/texreg/blob/roxygenize-extract/R/extract.R, 
# which is expected to be included in the later version of the texreg package
extract_brmsfit <- function (model,
                             use.HDI = TRUE,
                             level = 0.9,
                             include.random = TRUE,
                             include.rsquared = TRUE,
                             include.nobs = TRUE,
                             include.loo.ic = TRUE,
                             reloo = FALSE,
                             include.waic = TRUE,
                             ...) {
  sf <- summary(model, ...)$fixed
  coefnames <- rownames(sf)
  coefs <- sf[, 1]
  se <- sf[, 2]
  if (isTRUE(use.HDI)) {
    hdis <- coda::HPDinterval(brms::as.mcmc(model, prob = level, combine_chains = TRUE))
    hdis <- hdis[seq(1:length(coefnames)), ]
    ci.low = hdis[, "lower"]
    ci.up = hdis[, "upper"]
  } else { # default using 95% posterior quantiles from summary.brmsfit
    ci.low = sf[, 3]
    ci.up = sf[, 4]
  }
  
  gof <- numeric()
  gof.names <- character()
  gof.decimal <- logical()
  if (isTRUE(include.random) & isFALSE(!nrow(model$ranef))) {
    sr <- summary(model, ...)$random
    sd.names <- character()
    sd.values <- numeric()
    for (i in 1:length(sr)) {
      sd <- sr[[i]][, 1]
      sd.names <- c(sd.names, paste0("SD: ", names(sr)[[i]], names(sd)))
      sd.values <- c(sd.values, sd)
    }
    gof <- c(gof, sd.values)
    gof.names <- c(gof.names, sd.names)
    gof.decimal <- c(gof.decimal, rep(TRUE, length(sd.values)))
  }
  if (isTRUE(include.rsquared)) {
    rs <- brms::bayes_R2(model)[1]
    gof <- c(gof, rs)
    gof.names <- c(gof.names, "R$^2$")
    gof.decimal <- c(gof.decimal, TRUE)
  }
  if (isTRUE(include.nobs)) {
    n <- stats::nobs(model)
    gof <- c(gof, n)
    gof.names <- c(gof.names, "Num. obs.")
    gof.decimal <- c(gof.decimal, FALSE)
  }
  if (isTRUE(include.loo.ic)) {
    looic <- brms::loo(model, reloo = reloo)$estimates["looic", "Estimate"]
    gof <- c(gof, looic)
    gof.names <- c(gof.names, "loo IC")
    gof.decimal <- c(gof.decimal, TRUE)
  }
  if (isTRUE(include.waic)) {
    waic <- brms::waic(model)$estimates["waic", "Estimate"]
    gof <- c(gof, waic)
    gof.names <- c(gof.names, "WAIC")
    gof.decimal <- c(gof.decimal, TRUE)
  }
  
  tr <- texreg::createTexreg(coef.names = coefnames,
                             coef = coefs,
                             se = se,
                             ci.low = ci.low,
                             ci.up = ci.up,
                             gof.names = gof.names,
                             gof = gof,
                             gof.decimal = gof.decimal)
  return(tr)
}