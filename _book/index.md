--- 
title: "Course Handouts for Bayesian Data Analysis Class"
author: "Mark Lai"
date: "2019-12-14"
bibliography:
- ../notes/references.bib
- packages.bib
description: This is a collection of my course handouts for PSYC 621 class in 
  the 2019 Spring semester. Please contact me [mailto:hokchiol@usc.edu] for any
  errors (as I'm sure there are plenty of them). 
documentclass: book
link-citations: yes
site: bookdown::bookdown_site
biblio-style: apalike
header-includes: 
    - \usepackage{amsmath}
---

\DeclareMathOperator{\Cov}{Cov}
\DeclareMathOperator{\Tr}{Tr}
\newcommand{\diag}{\operatorname{diag}}
\newcommand{\logit}{\operatorname{logit}}
\newcommand{\logistic}{\operatorname{logistic}}
\newcommand{\E}{\mathrm{E}}
\newcommand{\bv}[1]{\boldsymbol{\mathbf{#1}}}
\newcommand{\SD}{\mathit{SD}}
\newcommand{\Var}{\mathrm{Var}}
\newcommand{\SDp}{\SD_\text{post}}
\newcommand{\SE}{\mathit{SE}}
\newcommand{\hSE}{\widehat{\mathit{SE}}}
\newcommand{\MSE}{\mathit{MSE}}
\newcommand{\RMSE}{\mathit{RMSE}}
\newcommand{\Deff}{\mathit{Deff}}
\newcommand{\df}{\mathit{df}}
\newcommand{\RE}{\mathit{RE}}
\newcommand{\DKL}{D_\textrm{KL}}
\newcommand{\dd}{\; \mathrm{d}}
\newcommand{\norm}{\mathcal{N}}

# Preface {-}

This is a collection of my course handouts for PSYC 621 class. The materials are
based on the book by @mcelreath2016statistical, the `brms` package
[@Burkner2017], and the [STAN language](https://mc-stan.org/). Please [contact
me](mailto:hokchiol@usc.edu) for any errors (as I'm sure there are plenty of
them).