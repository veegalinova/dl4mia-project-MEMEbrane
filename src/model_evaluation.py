#======================================================================================================
#======================================================================================================
#======================================================================================================
#EVALUATION OF THE TRAINED MODEL
#======================================================================================================
#======================================================================================================
#======================================================================================================


#upload validation dataset and generate STD from the binary mask
validation_data = SDTDataset(filepath) #TODO adjust

#convert to Torch data
batch_size = 1
shuffle = False
workers = 8
validation_data = DataLoader(validation_data, batch_size=batch_size, shuffle = shuffle, num_workers = workers)

#set unet to evaluation mode
unet.eval() #TODO change the name of the unet if needed



(precision_list,recall_list,accuracy_list,) = ([],[],[],)

#iterate over evaluation images
for idx, (image, mask, sdt) in enumerate(tqdm(val_dataloader)):

    #retrieve image
    image = image.to(device)

    #generate prediction from neural network
    pred = unet(image)

    #removes redundant dimensions I think?
    image = np.squeeze(image.cpu())
    gt_labels = np.squeeze(mask.cpu().numpy())
    pred = np.squeeze(pred.cpu().detach().numpy())

    
