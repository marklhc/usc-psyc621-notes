--- 
title: "Course Handouts for Bayesian Data Analysis Class"
author: "Mark Lai"
date: "2020-06-04"
bibliography:
- ../notes/references.bib
- packages.bib
description: This is a collection of my course handouts for PSYC 621 class in 
  the 2019 Fall semester. Please contact me [mailto:hokchiol@usc.edu] for any
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

You can download the EPUB version of the notes with the "Download" button. 

In some chapters I sourced some extra R codes. They can be found [here](https://github.com/marklhc/usc-psyc621-notes). 

The notes were last built with:


```r
sessioninfo::session_info()
```

```
## ─ Session info ───────────────────────────────────────────────────────────────
##  setting  value                       
##  version  R version 3.6.3 (2020-02-29)
##  os       Ubuntu 18.04.4 LTS          
##  system   x86_64, linux-gnu           
##  ui       X11                         
##  language (EN)                        
##  collate  en_US.UTF-8                 
##  ctype    en_US.UTF-8                 
##  tz       America/Los_Angeles         
##  date     2020-06-04                  
## 
## ─ Packages ───────────────────────────────────────────────────────────────────
##  package     * version date       lib source        
##  assertthat    0.2.1   2019-03-21 [1] CRAN (R 3.6.0)
##  bookdown      0.19    2020-05-15 [1] CRAN (R 3.6.3)
##  cli           2.0.2   2020-02-28 [1] CRAN (R 3.6.2)
##  crayon        1.3.4   2017-09-16 [1] CRAN (R 3.6.3)
##  digest        0.6.25  2020-02-23 [1] CRAN (R 3.6.3)
##  evaluate      0.14    2019-05-28 [1] CRAN (R 3.6.3)
##  fansi         0.4.1   2020-01-08 [1] CRAN (R 3.6.2)
##  glue          1.4.1   2020-05-13 [1] CRAN (R 3.6.3)
##  htmltools     0.4.0   2019-10-04 [1] CRAN (R 3.6.1)
##  knitr         1.28    2020-02-06 [1] CRAN (R 3.6.2)
##  magrittr      1.5     2014-11-22 [1] CRAN (R 3.6.0)
##  Rcpp          1.0.4.6 2020-04-09 [1] CRAN (R 3.6.3)
##  rlang         0.4.6   2020-05-02 [1] CRAN (R 3.6.3)
##  rmarkdown     2.2     2020-05-31 [1] CRAN (R 3.6.3)
##  sessioninfo   1.1.1   2018-11-05 [1] CRAN (R 3.6.0)
##  stringi       1.4.6   2020-02-17 [1] CRAN (R 3.6.2)
##  stringr       1.4.0   2019-02-10 [1] CRAN (R 3.6.0)
##  withr         2.2.0   2020-04-20 [1] CRAN (R 3.6.3)
##  xfun          0.14    2020-05-20 [1] CRAN (R 3.6.3)
##  yaml          2.2.1   2020-02-01 [1] CRAN (R 3.6.2)
## 
## [1] /home/markl/R/x86_64-pc-linux-gnu-library/3.6
## [2] /usr/local/lib/R/site-library
## [3] /usr/lib/R/site-library
## [4] /usr/lib/R/library
```
