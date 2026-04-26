
<!-- README.md is generated from README.Rmd. Please edit that file -->

# Gmi

<!-- badges: start -->

<!-- badges: end -->

Gmi: Group-Level Main Effects and Interactions in High-Dimensional Data.

Gmi is a package for analyzing the high dimensional data with
group-level main and interaction effetcs, developed by the Huazhen Lin’s
lab.

## Installation

You can install the development version of Gmi from
[GitHub](https://github.com/), firstly, install the ‘remotes’ package.

``` r
install.packages("remotes")
remotes::install_github("Moyu-nie/Gmi")
```

## Example

This is a basic example which shows you how to solve a common problem:

### Generate data

``` r
library(Gmi)
p <- 100
n <- 500
alpha2 <- c(rep(c(4,3,2),each = 2),rep(0,p-3*2))
gamma2 <- rep(0,choose(p,2))
mu2 <- 2 # intercept
rho2 <- 0.3 # correlation 

# interact between group1 and group2
aall = paste("X", combn(1 : p, 2, paste, collapse="X"), sep="")
aa1 = outer(1:2, 3:4, f <- function(x, y) {
  paste("X", pmin(x, y), "X", pmax(x, y), sep = "")
})
aa1 = as.vector(aa1)
gamma2[match(aa1, aall)] <- c(rep(4,4))

datList <- gendata(1, n ,p , alpha2, gamma2, mu2, rho2)
x2 = datList$x
Y2 = datList$Y
```

### Estimation

``` r
# fit
Gmi.fit = Gmi(x2, Y2, beta = rep(0,p), lambda.min.ratio = 0.02, n.lambda = 100, penalty = "SCAD", eta = 0.6, tune = "EBIC")
#> the method is finished
# estimate main effects
print(Gmi.fit$beta.m)
#> [1] 4.216678 4.216198 2.951518 2.952456 2.276872 2.276319
# estimate interaction effects
print(Gmi.fit$beta.i)
#> [1] 4.321111 4.422514 4.423419 4.425062
```
