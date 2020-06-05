pbma_brm_lm <- function(...) {
  # Perform model averaging using pseudo BMA with Bayesian bootstrap for 
  # `brmsfit` objects
  dots <- list(...)
  nsims <- vapply(dots, function(x) x$fit@sim$iter, FUN.VALUE = numeric(1))
  if (min(nsims) != max(nsims)) stop("All models need to have same number of draws.\n")
  model_names <- sapply(match.call(expand.dots = FALSE)$..., deparse)
  names(dots) <- model_names
  pbma_weight <- brms::loo_model_weights(..., method = "pseudobma")
  draws <- lapply(dots, fixef, summary = FALSE)
  par_names <- unique(unlist(lapply(draws, colnames)))
  niter <- nrow(draws[[1]])
  npar <- length(par_names)
  out <- matrix(0, nrow = niter, ncol = npar)
  colnames(out) <- par_names
  draws_full <- lapply(seq_along(draws), function(i) {
    draw <- draws[[i]]
    out[ , colnames(draw)] <- draw * pbma_weight[i]
    out
  })
  draws_pbma <- Reduce(`+`, draws_full)
  likelihoods <- lapply(seq_along(draws), function(i) {
    exp(log_lik(dots[[i]]) + log(pbma_weight[i]))
  })
  loglik_pbma <- log(Reduce(`+`, likelihoods))
  # loo_list$pbma <- loo::loo(loglik_pbma)
  # out <- loo::compare(x = loo_list)
  print(pbma_weight)
  invisible(draws_pbma)
}
