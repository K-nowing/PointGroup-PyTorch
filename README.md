# PointGroup-PyTorch
PointGroup: Dual-Set Point Grouping for 3D Instance Segmentation (CVPR2020)

**Reimplementation of PointGroup with different backbone(KPConv, DGCNN) for study purposes.**

[Reimplementation Process Post](https://knowing.tistory.com/25)



## Model Architecture
I used KPConv for Backbone, DGCNN for ScoreNet
<img src="https://user-images.githubusercontent.com/78612464/134359495-0eede73f-c555-47c6-9ffc-2e047eaacc95.png" width="80%">
<p align="center">
<img src="https://user-images.githubusercontent.com/78612464/134362887-18af062a-5398-4677-930b-fed6a72053d0.png" width="60%">
</p>

## Inference Result
### Semantic segmentation
<p>
<img src="https://user-images.githubusercontent.com/78612464/134514282-bae3118c-6391-4f78-ab25-ca0fff54f792.png">
</p>



### Instance segmentation
<p align="center">
<img src="https://user-images.githubusercontent.com/78612464/134513549-4aa52b03-da11-4112-be0f-e31ba38ae2e3.png" width="40%">
</p>

<p>
<img src="https://user-images.githubusercontent.com/78612464/134514454-37f6f70d-fdf9-455a-b9de-9c4ca8b4830d.png">
</p>

## Training

* [Data Preparation](./doc/S3DIS_data.md)

Simply run the following script to start the training:

        python train.py
        

## Testing

Simply run the following script to start the testing(recommended to run only 1 epoch):

        python test.py   # you should change weight directory


        
## Reference
1. [KPConv-PyTorch](https://github.com/HuguesTHOMAS/KPConv-PyTorch)
2. [dgcnn.pytorch](https://github.com/AnTao97/dgcnn.pytorch)
3. visualizing with [pptk](https://github.com/heremaps/pptk)
