#' Group-level Main Effects and Interactions for High-Dimensional Generalized Linear Models
#'
#' @useDynLib Gmi, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @importFrom stats rbinom
#' @importFrom utils combn
#' @param X A numeric matrix of predictors with dimensions n × p.
#' @param y A numeric response vector of length n.
#' @param beta Initial values for regression coefficients. If NULL, initialized to zero.
#' @param penalty penalty Choose from \code{LASSO}, \code{SCAD} and \code{MCP}. Default
#' is 'SCAD'.
#' @param penalty.factor A multiplicative factor for the penalty applied to each coefficient. If supplied,
#'  penalty.factor must be a numeric vector of length equal to the number of columns of X. The purpose of
#'  penalty.factor is to apply differential penalization if some coefficients are thought to be more likely
#'  than others to be in the model. In particular, penalty.factor can be 0, in which case the coefficient is
#'  always in the model without shrinkage.
#' @param inter.penalty.factor The penalty factor for interactions effects. Default is 1.
#' @param lam.list A user supplied \eqn{\lambda} sequence.
#' @param lambda.min.ratio Optional input. smallest value for \code{lambda}, as
#' a fraction of \code{max.lam}, the (data derived) entry value. the default
#' depends on the sample size \code{n} relative to the number of variables
#' @param max.iter maximum number of iteration in the computation. Default is
#' 100.
#' @param n.lambda The number of \code{lambda} values. Default is 100.
#' @param eps Tolerance threshold. Coefficients below this threshold
#'   are treated as zero.
#' @param tune Tuning parameter selection method.
#' 'AIC', 'BIC', 'EBIC' and 'GIC' are available options. Default is EBIC.
#' @param ebic.gamma the gamma parameter value in the EBIC criteria. Default is
#'   1.
#' @param eta A scalar in between 0 and 1 that controls the proportion of \code{lambda} allocated to sparsity (lasso-type) versus fusion penalty.
#'
#' @returns An object with Gmi.
#' \describe{
#'   \item{beta}{Final coefficient vector including both main effects and interactions.}
#'   \item{a_0}{Intercept corresponding to the selected model.}
#'   \item{a_0.list}{Intercepts along the lambda path.}
#'   \item{beta.m.mat}{Matrix of main effect coefficients for each lambda.}
#'   \item{beta.i.mat}{List of interaction coefficients across lambda values.}
#'   \item{beta.m}{Selected main effect coefficients.}
#'   \item{beta.i}{Selected interaction coefficients.}
#'   \item{lambda}{Sequence of lambda values used.}
#'   \item{mainInd.list}{List of indices of selected main effects for each lambda.}
#'   \item{mainInd}{Indices of main effects in the selected model.}
#'   \item{cri.list}{Values of the selected criterion across lambdas.}
#'   \item{loglik.list}{Log-likelihood values across lambdas.}
#'   \item{cri.loc}{Index of the lambda minimizing the selected criterion.}
#'   \item{all.locs}{Indices of lambda minimizing AIC, BIC, EBIC, GIC, and MBIC respectively.}
#'   \item{interInd.list}{List of selected interaction term names for each lambda.}
#'   \item{interInd}{Names of selected interactions in the final model.}
#'   \item{Klist}{Number of estimated groups for each lambda.}
#' }
#' @export
Gmi <- function(X, y, beta, penalty = "lasso",
                penalty.factor = rep(1, ncol(X)),
                inter.penalty.factor = 1,
                lam.list,
                lambda.min.ratio = 0.001,
                max.iter = 50, n.lambda = 100,
                eps = 1e-15, tune = "EBIC",
                ebic.gamma = 1, eta = 0.5) {
  pentype <- penalty
  n <- dim(X)[1]
  p <- dim(X)[2]

  if (missing(lambda.min.ratio)) {
    lambda.min.ratio <- ifelse(n < p, 0.01, 1e-04)
  }
  ################ prepare variables

  lambda <- NULL
  # storing the beta coefficents along the path
  beta.mat <- NULL
  if (is.null(beta)) {
    # the current beta estimates excluding the interation terms
    beta <- rep(0, p)
  }
  index <- NULL

  ################ save the original X for generating interactions
  X0 <- X

  ################ standardize design matrix

  ############# lambda list
  max.lam <- max(abs((1 / n) * (t(X) %*% (y - mean(y))))) / eta
  a_0list <- rep(log(mean(y) / (1 - mean(y))), n.lambda)

  if (missing(lam.list)) {
    min.lam <- max.lam * lambda.min.ratio
    lam.list <- exp(seq(from = log(max.lam), to = log(min.lam), length.out = n.lambda))
  } else {
    lam.list <- lam.list[lam.list <= max.lam]
    n.lambda <- length(lam.list)
  }
  ####### initialie
  a_0 <- a_0list[1]
  loglik.list <- cri.list <- AIC.list <- BIC.list <- MBIC.list <- EBIC.list <- GIC.list <-
    df.list <- df.m.list <- df.i.list <- Klist <- vector("numeric", n.lambda)
  ind.list.inter <- ind.list.main <- ind.list.inter.xlab <- beta.list.inter <-
    vector("list", n.lambda)
  ###### main part
  ## expand X matrix dynamically each step by incorporating the new candidates of
  ## interactions, which are the interactions of the current active variables minus
  ## the current interaction effects, the interactions will play the same role as
  ## the main effect, except we keep track of the list through the lambda sequence.
  ## Keep track of other indices as well.colnames(X)

  colnames(X) <- paste("X", 1:p, sep = "")
  ### heredity force
  nonPen <- rep(0, p)
  ### selected main effects from last step
  for (k in 1:n.lambda) {
    nonPen <- rep(0, p)
    lam <- lam.list[k]
    ### selected main effects from last step
    index <- which(abs(beta[1:p]) > eps)
    if (length(index) > 1) {
      ### find the candidate interation effects following strong heredity strong heredity
      aa <- paste("X", combn(index, 2, paste, collapse = "X"), sep = "")
      aa <- as.vector(aa)
      newinter <- t(apply(X0[, index], 1, combn, 2, prod))

      group <- Sepgroup(beta[1:p][index], 0.1)
      ### another constraint
      bb <- c()
      if (length(group) > 1) {
        for (i in 1:(length(group) - 1)) {
          for (j in ((i + 1):length(group))) {
            loc_sep1 <- match(beta[1:p][index][group[[i]]], beta[1:p])
            loc_sep2 <- match(beta[1:p][index][group[[j]]], beta[1:p])
            bb <- append(bb, as.vector(outer(loc_sep1, loc_sep2, function(x, y) {
              paste("X", pmin(x, y), "X", pmax(x, y), sep = "")
            })))
          }
        }
      } else {
        bb <- NULL
      }

      loc <- match(bb, aa)
      newinter <- newinter[, loc]
      curInter <- colnames(X)[-(1:p)]
      ### cut curInter
      nonInter <- setdiff(curInter, bb)
      if (length(nonInter) != 0) {
        nonloc <- match(nonInter, colnames(X))
        X <- X[, -nonloc]
        beta <- beta[-nonloc]
      }

      curInter <- colnames(X)[-(1:p)]

      candInter <- setdiff(bb, curInter)
      curloc <- match(candInter, bb)
      if (length(index) + length(curloc) > 5e5) {
        k <- k - 1
        break
      }
      newinter <- as.matrix(newinter)
      newinter <- newinter[, curloc]

      ncurInter <- length(curInter)
      ncandInter <- length(candInter)

      if (ncurInter > 0) {
        ## active interaction terms, setting the penalty for the parents to 0
        for (indInter in 1:ncurInter) {
          pair <- as.numeric(strsplit(curInter[indInter], "X")[[1]][2:3])
          nonPen[pair[1]] <- 1
          nonPen[pair[2]] <- 1
        }
      }

      if (ncandInter > 0) {
        xnewname <- c(colnames(X), candInter)
        # ordering colnames
        re_order <- match(c(colnames(X)[1:p], bb), xnewname)
        xnewname <- xnewname[re_order]
        tmp <- newinter
        X <- cbind(X, tmp)
        # ordering X
        X <- X[, re_order]
        colnames(X) <- xnewname

        # expand the beta coefficent vector
        beta <- c(beta, rep(0, ncandInter))
        beta <- beta[re_order]
        # to include the candiate interaction terms.
      }
    }
    pf <- c(penalty.factor, rep(inter.penalty.factor, ncol(X) - p))
    nonpenind <- which(nonPen != 0)
    pf[nonpenind] <- 0
    ### update beta(expand)
    for (ite in 1:max.iter) {
      if (ite == 1) {
        cd.temp1 <- SBfusedlasso(
          X = X, y = y, a_0 = a_0, beta = beta, lam = lam,
          eta = eta, rho1 = 1, rho2 = 1, penalty.type = pentype,
          epsilon1 = 1e-7, epsilon2 = 1e-7, maxiter1 = 100, maxiter2 = 20, pf = pf
        )
        a_0 <- cd.temp1$a_0
        beta <- cd.temp1$beta
        ind1 <- which(abs(beta) > 1e-10)
      }
      if ((ite > 1) && (length(ind1) >= 2)) {
        cd.temp2 <- SBfusedlasso(
          X = X[, ind1], y = y, a_0 = a_0, beta = beta[ind1], lam = lam, eta = eta, rho1 = 1, rho2 = 1, penalty.type = pentype,
          epsilon1 = 1e-7, epsilon2 = 1e-7, maxiter1 = 100, maxiter2 = 20, pf = pf[ind1]
        )
        a_0 <- cd.temp2$a_0
        beta2 <- beta
        beta2[ind1] <- cd.temp2$beta
        ######## redetect active set check.beta=new.beta
        cd.temp3 <- SBfusedlasso(
          X = X, y = y, a_0 = a_0, beta = beta2, lam = lam, eta = eta, rho1 = 1, rho2 = 1, penalty.type = pentype,
          epsilon1 = 1e-7, epsilon2 = 1e-7, maxiter1 = 100, maxiter2 = 20, pf = pf
        )
        a_0 <- cd.temp3$a_0
        beta <- cd.temp3$beta
        ind3 <- which(abs(beta) > 1e-10)
        if (setequal(ind1, ind3)) {
          break
        }
        ind1 <- ind3
      }
    }
    # record the main and interaction
    ind.list.main[[k]] <- which(abs(beta[1:p]) > 1e-10)
    ind.list.inter[[k]] <- which(abs(beta[-(1:p)]) > 1e-10)
    size.main <- length(ind.list.main[[k]])
    size.inter <- length(ind.list.inter[[k]])
    index <- which(abs(beta) > 1e-10)
    beta[-index] <- 0
    # hier = "strong" is default
    if (size.inter > 0) {
      ### if interaction effects are detected, enforce the strong heredity in this step
      tmpindname <- colnames(X)[ind.list.inter[[k]] + p]
      cur.main.ind <- intertomain(tmpindname)
      ind.list.main[[k]] <- union(ind.list.main[[k]], cur.main.ind)
      index <- union(index, cur.main.ind)
    }

    size.main <- length(ind.list.main[[k]])

    index <- sort(index)
    beta.n <- beta
    a_0.n <- a_0

    df.list[k] <- size.main + size.inter
    loglik.list[k] <- loglikbinomial(cbind(1, X), y, c(a_0.n, beta.n))

    if (length(beta) > p) {
      tmp <- which(abs(beta[-(1:p)]) > 1e-10)
      beta <- beta[c(1:p, p + tmp)]
      beta.n <- beta.n[c(1:p, p + tmp)]
      X <- X[, c(1:p, p + tmp)]
      if (length(tmp) > 0) {
        ## if interaction effects are selected.
        ind.list.inter.xlab[[k]] <- colnames(X)[-(1:p)]
        beta.list.inter[[k]] <- beta.n[-(1:p)]
      }
    }
    ### binomial is default
    if (max(abs(beta)) > 10) {
      k <- k - 1
      break
    }
    beta.s <- beta.n
    a_0.s <- a_0.n
    a_0list[k] <- a_0.s

    beta.mat <- cbind(beta.mat, beta.s[1:p])

    df.m.list[k] <- length(ind.list.main[[k]])
    df.i.list[k] <- length(beta) - p
    if (df.list[k] >= n - 1) {
      break
    }
    ### 保存组数信息
    Klist[k] <- length(Sepgroup(beta[1:p], 0.1)) + length(Sepgroup(beta[-(1:p)], 0.1))
  }
  ### hier is strong
  p.eff <- p + df.m.list * (df.m.list - 1) / 2
  AIC.list <- loglik.list + 2 * df.list
  BIC.list <- loglik.list + log(n) * df.list
  MBIC.list <- loglik.list + log(n) * df.list + 2 * ebic.gamma *
    log(choose(p.eff, Klist))
  EBIC.list <- loglik.list + log(n) * df.list + 2 * ebic.gamma *
    log(choose(p.eff, df.list))

  GIC.list <- loglik.list + log(log(n)) * log(p.eff) * df.list
  cri.list <- switch(tune,
    AIC = AIC.list,
    BIC = BIC.list,
    EBIC = EBIC.list,
    GIC = GIC.list,
    MBIC = MBIC.list
  )

  region <- which(df.list[1:k] < 2 * n)
  cri.loc <- which.min(cri.list[region])
  AIC.loc <- which.min(AIC.list[region])
  BIC.loc <- which.min(BIC.list[region])
  MBIC.loc <- which.min(MBIC.list[region])
  EBIC.loc <- which.min(EBIC.list[region])
  GIC.loc <- which.min(GIC.list[region])

  all.locs <- c(AIC.loc, BIC.loc, EBIC.loc, GIC.loc, MBIC.loc)
  lambda <- lam.list[seq_len(ncol(beta.mat))]


  if (length(ind.list.inter) == 0) {
    interInd <- NULL
  } else {
    interInd <- ind.list.inter.xlab[[cri.loc]]
  }
  if (length(beta.list.inter) == 0) {
    beta.i <- NULL
  } else {
    beta.i <- beta.list.inter[[cri.loc]]
  }
  message("the method is finished")

  val <- list(
    beta = beta, a_0 = a_0list[cri.loc], a_0.list = a_0list, beta.m.mat = beta.mat, beta.i.mat = beta.list.inter, beta.m = beta.mat[ind.list.main[[cri.loc]], cri.loc], beta.i = beta.i, lambda = lambda[1:k], mainInd.list = ind.list.main[1:k],
    mainInd = ind.list.main[[cri.loc]], cri.list = cri.list[1:k], loglik.list = loglik.list[1:k],
    cri.loc = cri.loc, all.locs = all.locs, interInd.list = ind.list.inter.xlab[1:k], interInd = interInd, Klist = Klist
  )
  return(val)
}
