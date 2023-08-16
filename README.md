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


then it gave an error again so i mad a change iin dataloader:
def get_dataloader(
    dataset: torch.utils.data.Dataset,
    path_to_csv: str,
    phase: str,
    batch_size: int = 1,
    num_workers: int = 2,
    json_file: str = None,
    fold: int = 0,
    train_dir: str = Config.newGlobalConfigs.train_root_dir,
    test_dir: str = Config.newGlobalConfigs.test_root_dir,
    is_process_mask: bool = True):

    '''
    Load the dataset into batches from the disk.

    Parameters
    ----------
    dataset: torch.utils.data.Dataset (a datset instance)
    path_to_csv: str (path to a xlsx or csv file)
    phase: str (training, validation, or test phase)
    batch_size: int (loader batch size to be specified)
    num_workers: int (number of workers to be employed to load data)
    json_file: str (path to dataset split json file)
    fold: int (fold index to be specified for validation phase)
    train_dir: str (training directory)
    test_dir: str(test directory)
    is_process_mask: bool (whether to stack tumor classes to the original label)


    Returns:
    -------
    dataloader: torch.utils.data.DataLoader (loader object)
    '''
    # get the dataframe with full path to patients records
    df = insert_cases_paths_to_df(path_to_csv, train_dir = train_dir, 
                                  test_dir= test_dir, json_file=json_file, fold=fold)

    # seperate training data frame from validation data frame
    train_df = df.loc[df['phase'] == 'train'].reset_index(drop=True)
    val_df = df.loc[df['phase'] == 'val'].reset_index(drop=True)
    df = train_df if phase == "train" else val_df
    logger.info(" {} phase selected.".format(phase.capitalize()))

    if df.empty:
        logger.info("The DataFrame is empty.")
    else:
        logger.info(" First row in the DataFrame: {}".format(df.iloc[0, :].tolist()))

    # Load the dataset for the phase specified
    dataset = dataset(df, phase, is_process_mask = is_process_mask)
    logger.info(f' Total examples in the {phase} dataset: {len(dataset)}')

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        drop_last= True)
    return dataloader
