# Bioinfo_material

This repository contains a collection of bioinformatics functions used to analyse bulk/single-cell transcriptomics data. <br>
The `src` folder contains all the relevant scripts/functions where;
- `network` contains network analysis methods to study co-expression networks (_motifs detection_, _clustering coefficient_, etc...)
* `bioinfo` contains general methods for differential expression, pathway enrichment analysis, and other methods
    - `run_pathifier` function is a Python implementation of the _Pathifier_ ([download here](http://www.bioconductor.org/packages/release/bioc/html/pathifier.html))
      method developed by [Drier et al.](https://www.pnas.org/content/110/16/6388) <br>
      This method allows to compress gene-based info into pathway-based info. <br />
      This implementation allows to choose between the original method and the [_PathTracer_](https://www.nature.com/articles/s41598-019-52529-3) method. <br>
      Also, this is a modified version of both: the optimal number of PCs to use for the _Principal Curve_ computation is given by a permutation test on the components: <br /> 
      &nbsp;&nbsp; - see `my_PCA.select_components_above_background` for reference. <br>
      &nbsp;&nbsp; - an example of the pipeline is given in the module `pathifier_example_structure.py`
      
