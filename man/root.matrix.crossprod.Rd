\name{root.matrix.crossprod}
\alias{root.matrix.crossprod}
\title{Root of X^TX}
\description{Computes the root of the Gramian X^TX.}
\usage{
root.matrix.crossprod(X)
}
\arguments{
  \item{X}{a matrix}
}

\value{a symmetric matrix \code{V} where \code{V^2} equals \code{X^TX}}

\examples{
set.seed(1)
n <- 100
p <- 3
X <- matrix(rnorm(n*p),nrow=n, ncol=p)
test <- root.matrix.crossprod(X)
## control results
t(X) \%*\% X
test \%*\% test
}
\keyword{algebra}
