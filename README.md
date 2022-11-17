# HDM
Learning Hierarchical Dynamics with Spatial Adjacency for Image Enhancement (Accepted with ACMMM2022)


### Dependencies and Installation

* python3.6
* PyTorch==1.4
* NVIDIA GPU+CUDA
* numpy
* opencv-python
* dcnv2
* https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch.git you need match it with pytorch1.4



### Datasets Preparation
you need prepare SOT indoor DataSet


### Test
<pre><code>
python test_hdm_sots.py
</code></pre>


## Citation
The paper can be accessed from https://dl.acm.org/doi/abs/10.1145/3503161.3548322 .</br>
If you find our work useful in your research, please cite:
<pre><code>
@inproceedings{10.1145/3503161.3548322,
        author = {Liang, Yudong and Wang, Bin and Ren, Wenqi and Liu, Jiaying and Wang, Wenjian and Zuo, Wangmeng},
        title = {Learning Hierarchical Dynamics with Spatial Adjacency for Image Enhancement},
        year = {2022},
        isbn = {9781450392037},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        url = {https://doi.org/10.1145/3503161.3548322},
        doi = {10.1145/3503161.3548322},
        abstract = {In various real-world image enhancement applications, the degradations are always non-uniform or non-homogeneous and diverse, which challenges most deep networks with fixed parameters during the inference phase. Inspired by the dynamic deep networks that adapt the model structures or parameters conditioned on the inputs, we propose a DCP-guided hierarchical dynamic mechanism for image enhancement to adapt the model parameters and features from local to global as well as to keep spatial adjacency within the region. Specifically, channel-spatial-level, structure-level, and region-level dynamic components are sequentially applied. Channel-spatial-level dynamics obtain channel- and spatial-wise representation variations, and structure-level dynamics enable modeling geometric transformations and augment sampling locations for the varying local features to better describe the structures. In addition, a novel region-level dynamic is proposed to generate spatially continuous masks for dynamic features which capitalizes on the Dark Channel Priors (DCP). The proposed region-level dynamics benefit from exploiting the statistical differences between distorted and undistorted images. Moreover, the DCP-guided region generations are inherently spatial coherent which facilitates capturing local coherence of the images. The proposed method achieves state-of-the-art performance and generates visually pleasing images for multiple enhancement tasks,i.e. , image dehazing, image deraining and low-light image enhancement. The codes are available at https://github.com/DongLiangSXU/HDM.},
        booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
        pages = {2767–2776},
        numpages = {10},
        keywords = {depth, dark channel priors, region-level dynamics, hierarchical dynamics, spatial adjacency, image enhancement},
        location = {Lisboa, Portugal},
        series = {MM '22}
}
</code></pre>

## Contact Us
If you have any questions, please contact us:</p>
liangyudong006@163.com </br>
202022407046@email.sxu.edu.cn
###

`