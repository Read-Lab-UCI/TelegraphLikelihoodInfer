# telegraph_likelihood_based_infer
 maximum likelihood inferenence of kinetic parameters of telegraph model by computing profile likelihood

 only input required is mRNA count matrix in csv file or processed distribution of distribution in shelve:
 ```
 python wrapper.py --counts mRNA_count_matrix_file.csv
 ```
 if mRNA count matrix is in the format of rows as sample, column as genes:
 ```
 python wrapper.py --counts mRNA_count_matrix_file.csv --transpose True
 ```
  or 
 ```
 python wrapper.py --pexp histogram_shelve(file name string, no file extension required)
 ```
 The pipeline has a default of max mRNA count of 298 so that a necessary size library is generated accordingly, in any case a larger max count is needed:
 ```
 python wrapper.py --counts mRNA_count_matrix_file.csv --transpose True --maxcount 1000
 ```
 This is would be a necessary argument if the mrna count data is considered to be downsampled. Then, the library maxcount would be estimated to be maxcount/downsample:
 ```
 python wrapper.py --counts mRNA_count_matrix_file.csv --transpose True --maxcount 500 --downsample 0.3
 ```
 By defaut, most genes has rather small expression that is less than 300, for genes with expression larger than maxcount, they will not be processed. For a downsample rate 
 of 0.3, the library max will be around 1000. 

 To compute mRNA distribution from kinetic parameter, the default parameters are ksyn, koff, kon, if one desires to use gene state percentage on instead of kon:
 ```
 python wrapper.py --counts mRNA_count_matrix_file.csv --percentage True
 ```
 if a specific library is to use, you can also provide the library file path, otherwise a default library will be used (will be generated in the first time this code is ran):
 ```
 python wrapper.py --counts mRNA_count_matrix_file.csv --psim path_to_library_shelve(file name string,no file extension required)
 ```
 To generate the library, one can specify the number of grid point in each parameter axis for ksyn, koff, kon, default is 60 by 60 by 60, or the maximum ksyn value in log10:
 ```
 python grid_library.py --shape 100 100 100 --ksyn_max 3.0
 ```
 The library can also include sensitivty and keep the transition matrix for further analysis:
 ```
 python grid_library.py --sensitivity True --transition True
 ```
 The workflow also can include different mRNA capture rate in sc-RNA seq data, but need to provide a specific capture rate, default is fully capture (1.0):
 ```
 python wrapper.py --counts mRNA_count_matrix_file.csv  --downsample 0.3
 ```
 For profile likelihood, a standard  cutoff of 0.95 confidence interval is used, if a specific cutoff is desired:
 ```
 python wrapper.py --counts mRNA_count_matrix_file.csv --cutoff 0.98
 ```
 For profile likelihood, the default way to compute the confidence interval is based on chi square value based on cutoff value for degree of freedom of 1, if instead would 
 like to used based on probability distribution of the parameter:
 ```
 python wrapper.py --counts mRNA_count_matrix_file.csv --probability True
 ```
 The pipeline can also sample from mRNA distribution to mimic influence of sampling noise in distribution, and the effect of sampling in inference. For example, sample 100 
 replicates from the mRNA distribution with specific number of cells (default 1000). This could be useful specifically when looking into the quality of inference when 
 ground truth is known:
 ```
 python wrapper.py --counts mRNA_count_matrix_file.csv --repeat 100 --cell 2000
 ```
 The pipeline is default to run parallel on all available cpu cores, if a desired number of cores to be used:
 ```
 python wrapper.py --counts mRNA_count_matrix_file.csv --parallel 20
 ```
 In the case of deploying in hpc, the pipeline is designed to run as a slurm array job, please run wrapper.sub with desired specification. This would break into batches 
 which run distributions in loop, for example starting of the 0th distribution, in a batch of 30:
 ```
 python wrapper.py --counts mRNA_count_matrix_file.csv --loop 1 --index 0 --end 30
 ```
 For self inference of the library, we recommand to run self inference on server or hpc:
 ```
 python wrapper.py --pexp library_name --self 1
 ```

<a href="https://doi.org/10.5281/zenodo.16915449"><img src="https://zenodo.org/badge/975005025.svg" alt="DOI"></a>
 
