\name{magnitude}
\alias{magnitude}
\alias{magnitude.breakpointsfull}
\title{Magnitudes of Breakpoints}
\description{
  Computes magnitude statistics for breakpoints.
}
\usage{
\method{magnitude}{breakpointsfull}(object,
  interval = 0.1, breaks = NULL, component = "trend", \dots)
}
\arguments{
  \item{object}{an object of class \code{"breakpointsfull"} as computed by
    \code{\link{breakpoints}}.}
  \item{interval}{the (one-side) interval over which to compute the magnitude for the columns
    \code{RMSD} and \code{MAD}. If below 1, fraction of the time series length,
    otherwise number of observations.}
  \item{breaks}{how many breaks to use or which statistic to use for estimating it,
    see \code{\link{breakpoints}} for details. If \code{NULL}, uses the defaults.}
  \item{component}{which covariate(s) used for fitting the \code{breakpointsfull}
    object to use for magnitude estimation. The components are additive.}
  \item{\dots}{\emph{ignored}}
}

\details{
  The breakpoint magnitude is estimated using several statistics:
  the difference (\emph{diff}) in the fitted value immediately \emph{before}
  and \emph{after} the break, and the root mean squared deviation (\emph{RMSD}),
  mean absolute deviation (\emph{MAD}) and mean deviation (\emph{MD}) between the
  fitted values of the model preceding the break extrapolated over \emph{interval}
  samples after the break, and vice versa. RMSD and MAD should be more robust
  estimators of magnitude compared to \emph{diff}.
}

\value{
  A list with the following elements, compatible with the magnitude format as output by \code{\link[bfast]{bfast}}:
  \describe{
    \item{Mag}{a matrix containing the magnitude estimates (in columns) for each breakpoint (in rows),}
    \item{m.x}{sample number at which the largest break was detected, twice}
    \item{y.x}{fitted values \emph{before} and \emph{after} the largest break}
    \item{Magnitude}{the maximum \emph{diff} magnitude of the largest break}
    \item{Time}{time of the largest break (same as \code{m.x}, but single value)}
  }
}

\seealso{\code{\link{breakpoints}}}

\examples{
data(Nile)

trend <- 1:length(Nile)
bp.nile <- breakpoints(Nile ~ trend)

# The trend component is default, set "component" to the
# name of your coviariate(s), if it is different.
magnitude(bp.nile)
}

\keyword{regression}
 
