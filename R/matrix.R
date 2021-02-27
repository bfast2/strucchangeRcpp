root.matrix <- function(X)
{
    if((ncol(X) == 1L)&&(nrow(X) == 1L)) return(sqrt(X))
  if (getOption("strucchange.use_armadillo", FALSE))
    return(.sc_cpp_rootmatrix(X))
  
  X.eigen <- eigen(X, symmetric=TRUE)
  if(any(X.eigen$values < 0)) stop("matrix is not positive semidefinite")
  sqomega <- sqrt(diag(X.eigen$values))
  V <- X.eigen$vectors
        V <- V %*% sqomega %*% t(V)
	dimnames(V) <- dimnames(X)
	return(V)
}

root.matrix.crossprod <- function(X)
{
  if (!getOption("strucchange.use_armadillo", FALSE))
    return(root.matrix(crossprod(X)))
  if ("matrix" %in% class(X)) {
    return(.sc_cpp_rootmatrix_cross(X))
  }
  return(.sc_cpp_rootmatrix_cross(as.matrix(X)))
}


solveCrossprod <- function(X, method = c("qr", "chol", "solve")) {
  switch(match.arg(method),
    "qr" = chol2inv(qr.R(qr(X))),
    "chol" = chol2inv(chol(crossprod(X))),
    "solve" = solve(crossprod(X)))
}


