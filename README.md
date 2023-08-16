# brats_swin_unetr_2023


after changing configs
a change in preprocess.py was madebecause of following error:
**error:**
Configuring...
Configured. Now Loading dataset...
Traceback (most recent call last):
  File "/content/drive/MyDrive/Brats-20-Tumors-segmentation/test.py", line 145, in <module>
    main()
  File "/content/drive/MyDrive/Brats-20-Tumors-segmentation/test.py", line 120, in main
    test_loader = get_dataloader(BraTSDataset,
  File "/content/drive/MyDrive/Brats-20-Tumors-segmentation/DataLoader/dataset.py", line 253, in get_dataloader
    df = insert_cases_paths_to_df(path_to_csv, train_dir = train_dir, 
  File "/content/drive/MyDrive/Brats-20-Tumors-segmentation/utils/preprocess.py", line 88, in insert_cases_paths_to_df
    paths.append(path)
UnboundLocalError: local variable 'path' referenced before assignment
**change:**
def insert_cases_paths_to_df(df:str, 
                             train_dir:str = None, 
                             test_dir:str = None, 
                             json_file:str = None, 
                             fold: int = 0):
    ...
    for _ , row in df.iterrows():
        id = row["BraTS2023"]
        path = None
        type = None
        if id in os.listdir(train_dir):
            path = train_dir + "/" + id
            if id in train:
                type = "train"
            elif id in val:
                type = "val"
        elif id in os.listdir(test_dir):
            path = test_dir + "/" + id
            type = "test"
        paths.append(path)
        phase.append(type)
    ...
