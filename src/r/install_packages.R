# install_packages.R
# Installs the R packages needed by titanic_model.R, pinned to exact versions.
# Runs once when the Docker image is built: Rscript install_packages.R

# 'remotes' provides install_version(), which installs a specific package version
install.packages("remotes")

# Packages and the exact versions to install
packages <- c(dplyr = "1.2.1")

for (pkg in names(packages)) {
  remotes::install_version(pkg, version = packages[[pkg]], upgrade = "never")
}

# Stop the build if any package is missing or has the wrong version
for (pkg in names(packages)) {
  installed <- as.character(packageVersion(pkg))
  if (installed != packages[[pkg]]) {
    stop(sprintf("%s %s installed, expected %s", pkg, installed, packages[[pkg]]))
  }
  cat(sprintf("Installed %s %s\n", pkg, installed))
}