
## S3DIS Data

### Data

```
├── PointGroup-PyTorch          # main repository
└── Data                        # Data directory
    └── S3DIS                
        └── PG_data            
            └── original_ply
                ├── Area_1.ply  
                ├── Area_2.ply
                ├── Area_3.ply
                ├── Area_4.ply
                ├── Area_5.ply
                └── Area_6.ply
```
if you want to place your data anywhere else, you just habe to change the variable 'self.path' of 'S3DISDataset' class ([here](https://github.com/K-nowing/PointGroup-PyTorch/blob/main/datasets/S3DIS.py#L88)).

S3DIS dataset can be downloaded <a href="https://goo.gl/forms/4SoGp4KtH1jfRqEj2">here (4.8 GB)</a>. 


