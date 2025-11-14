# -1) Run
# conda install r-base=4.3.1
# conda install gsl

# 0) Use an up-to-date repo (avoid MRAN, which was retired)
options(repos = c(CRAN = "https://cloud.r-project.org"))

# 1) Tools you’ll use to pin versions
install.packages("remotes")

# 2) Install dependency versions compatible with R 4.3.x
remotes::install_version("Matrix", version = "1.6-5")   # 4.3-compatible
remotes::install_version("mgcv",   version = "1.8-42")  # imports Matrix, R >= 3.6
remotes::install_version("glmnet", version = "4.1-10")  # depends Matrix >= 1.0-6
remotes::install_version("mboost", version = "2.9-11")  # imports Matrix, survival, ...

install.packages(c("docopt", "BiocManager"), repos='https://stat.ethz.ch/CRAN/')
BiocManager::install(c("graph", "RBGL", "ggm", "Rgraphviz"), quiet=TRUE, update=TRUE, ask=FALSE)

remotes::install_version("MASS", version = "7.3-60.0.1")

install.packages(c("momentchi2", "pcalg"), dependencies=TRUE, quiet=TRUE, repos='https://stat.ethz.ch/CRAN/')
remotes::install_version("kpcalg", version = "1.0.1")

library(remotes)
install_github("Diviyan-Kalainathan/RCIT")
install.packages("./R_archives/SID_1.0.tar.gz", repos=NULL, type="source")
