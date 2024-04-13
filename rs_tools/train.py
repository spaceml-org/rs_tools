# Check HYDRA config:
# Datamodule needs to contain 2 datasets
# dataloader:
#   _target_: rs_tools._src.datamodule.ITIDataModule
#   datasets_spec:
#     dataset_1:
#       bands: all
#     dataset_2:
#       bands: all
#     
#   include_coords: True
#   include_cloudmask: True
#   include_nanmask: True

#   datasets_split:
#     train:
#       years:
#       months:
#       days:
#     val: 
#       years:
#       months:
#       days: