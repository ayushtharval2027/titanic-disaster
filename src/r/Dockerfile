# Base image: R 4.6.1 from the Rocker project
FROM rocker/r-ver:4.6.1

# All following commands run inside /app in the container
WORKDIR /app

# Install packages first so this layer is cached between builds
COPY install_packages.R .
RUN Rscript install_packages.R

# Copy the script in after packages
COPY titanic_model.R .

# Tell the script where the data is mounted (see docker run -v)
ENV DATA_DIR=/data

# Run the script when the container starts
CMD ["Rscript", "titanic_model.R"]