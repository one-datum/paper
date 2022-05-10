# # User config
# configfile: "showyourwork.yml"

# # Import the showyourwork module
# module showyourwork:
#     snakefile:
#         "showyourwork/workflow/Snakefile"
#     config:
#         config

# # Use all default rules
# use rule * from showyourwork

# rule download_archive:
#     output:
#         "src/archive.zip"
#     shell:
#         "curl https://zenodo.org/record/5593748/files/archive.zip --output {output}"

# rule pipeline_products:
#     input:
#         "src/archive.zip"
#     output:
#         "src/archive/inference/inferred.fits.gz",
#         "src/archive/noise/process.fits",
#         "src/archive/simulations/inferred.fits.gz"
#     shell:
#         "unzip {input} -d src"
