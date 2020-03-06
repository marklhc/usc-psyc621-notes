



# Brief Introduction to STAN

The engine used for running the Bayesian analyses covered in this course is
`STAN`, as well as the  `rstan` package that allows it to interface with R.
`STAN` requires some programming from the users, but the benefit is that it
allows users to fit a lot of different kinds of models. The goal of this lecture
is not to make you an expert of `STAN`; I myself only have used maybe just 1 or
2% of the power of `STAN`. Instead, the goal is to give you a brief introduction
with some sample codes, so that you can study further by yourself, and estimate
models that no frequentist estimation exists yet.

## `STAN`

`STAN` (http://mc-stan.org/) is itself a programming language, just like R. 
Strictly speaking it is not only for Bayesian methods, as you can actually do 
penalized maximum likelihood and automatic differentiation; however, it is most
commonly used as an MCMC sampler for Bayesian analyses. It is written in C++,
which makes it much faster than R (R is actually quite slow as a computational
language). You can actually write a `STAN` program without calling R or other 
software, although eventually you may want to use statistical software to 
post-process the posterior samples after running MCMC. There are interfaces of
`STAN` for different programs, including R, Python, MATLAB, Julia, Stata, and
Mathematica, and for us we will be using the `RStan` interface. 

### `STAN` code

In `STAN`, you need to define a model using the `STAN` language. Below is 
an example for the Poisson model, which is saved with the file name 
`"poisson_model.stan"`.


```stan
data {
  int<lower=0> N;  // number of observations
  int<lower=0> y[N];  // data array (counts);
}
parameters {
  real log_lambda;  // log of rate parameter
}
model {
  y ~ poisson_log(log_lambda);
  // prior
  log_lambda ~ normal(0, 5);
}
generated quantities {
  real lambda = exp(log_lambda);
  int yrep[N];
  for (i in 1:N) {
    yrep[i] = poisson_log_rng(log_lambda);
  }
}
```

In `STAN`, anything after `//` denotes comments and will be ignored by the
program, and in each blocks (e.g., `data {}`) a statement needs to be ended by a
semicolon (`;`). There are several blocks in the above `STAN` code:

- `data`: The data for input for `STAN` is usually not only a data set, but 
include other information, including sample size, number of predictors, and 
prior scales. Each type of data has an input type, such as 
    * `int` = integer, 
    * `real` = numbers with decimal places, 
    * `matrix` = 2-dimensional data of real numbers, 
    * `vector` = 1-dimensional data of real numbers, and 
    * `array` = 1- to many-dimensional data. For example `y[N]` is a 
    one-dimensional array of integers. 
you can set the lower and upper bounds so that `STAN` can check the input data
- `parameters`: The parameters to be estimated
- `transformed parameters`: optional variables that are transformation of the 
model parameters. It is usually used for more advanced models to allow for 
more efficient MCMC sampling. 
- `model`: It includes definition of priors for each parameter, and the 
likelihood for the data. There are many possible distributions that can be used 
in `STAN`. 
- `generated quantities`: Any quantities that are not part of the model but 
can be computed from the parameters for every iteration. Examples include 
posterior generated samples, effect sizes, and log-likelihood (for fit 
computation). 

## `RStan`

`STAN` is written in C++, which is a compiled language. This is different from 
programs like R, which you can input a command and get results right away. In 
contrast, a `STAN` program needs to be converted to something that can be
executed in your computer. The benefit, however, is that the programs can be run
much faster after the compilation process. 

To feed data from R to `STAN`, and import output from `STAN` to R, you will use 
the `rstan` package
(https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started). Then, follow the
following steps:

We will continue with the red card example:


```r
redcard_dat <- readr::read_csv("../data/redcard_data.zip") %>% 
  group_by(player) %>% 
  summarise(rater_dark = (mean(rater1) + mean(rater2)) / 2, 
            yellowCards = sum(yellowCards), 
            redCards = sum(redCards))
```

### Assembling data list in R 

First, you need to assemble a list of data for `STAN` input, which should match
the specific `STAN` program. In the `STAN` program we define two components
(`N` and `y`) for data, so we need seven elements in an R list:


```r
pois_sdata <- list(
  N = nrow(redcard_dat),  # number of observations
  y = redcard_dat$yellowCards  # outcome variable (yellow card)
)
```

### Call `rstan`


```r
library(rstan)
```

```
># Loading required package: StanHeaders
```

```
># rstan (Version 2.19.3, GitRev: 2e1f913d3ca3)
```

```
># For execution on a local, multicore CPU with excess RAM we recommend calling
># options(mc.cores = parallel::detectCores()).
># To avoid recompilation of unchanged Stan programs, we recommend calling
># rstan_options(auto_write = TRUE)
```

```
># 
># Attaching package: 'rstan'
```

```
># The following object is masked from 'package:tidyr':
># 
>#     extract
```

```r
rstan_options(auto_write = TRUE)
```


```r
m3 <- stan(file = "../codes/poisson_model.stan", data = pois_sdata, 
           # Below are optional arguments
           iter = 2000, 
           chains = 4, 
           cores = min(parallel::detectCores(), 4))
```

### Summarize the results

After you call the `stan` function in R, it will compile the `STAN` program, 
which usually takes a minute or so. Then it starts sampling. You can now see
a summary of the results by printing the results:


```r
print(m3, pars = c("lambda", "log_lambda"))
```

```
># Inference for Stan model: poisson_model.
># 4 chains, each with iter=2000; warmup=1000; thin=1; 
># post-warmup draws per chain=1000, total post-warmup draws=4000.
># 
>#             mean se_mean   sd  2.5%   25%   50%   75% 97.5% n_eff Rhat
># lambda     27.67       0 0.12 27.42 27.59 27.67 27.75 27.89  1156    1
># log_lambda  3.32       0 0.00  3.31  3.32  3.32  3.32  3.33  1155    1
># 
># Samples were drawn using NUTS(diag_e) at Fri Mar  6 10:07:20 2020.
># For each parameter, n_eff is a crude measure of effective sample size,
># and Rhat is the potential scale reduction factor on split chains (at 
># convergence, Rhat=1).
```

And you can also use the `shinystan` package to visualize the results:


```r
shinystan::launch_shinystan(m3)
```

## Resources

`STAN` is extremely powerful and can fit almost any statistical models, but the 
price is that it takes more effort to code the model. To learn more about 
`STAN`, please check out http://mc-stan.org/documentation/ for the manual, 
examples of some common models, and case studies (which includes more complex
models like item response theory). See
https://cran.r-project.org/web/packages/rstan/vignettes/rstan.html for a 
vignettes for working with the `rstan` package. 

As you see, fitting simple models in `STAN` may sometimes be more work, but as
we go further we will use the `brms` program that simplify the process for many
commonly used models, such as regression and multilevel models. On the other
hand, for truly complex models, `STAN` is actually a lifesaver as it would be
extremely hard to fit some of those models with other approaches.
