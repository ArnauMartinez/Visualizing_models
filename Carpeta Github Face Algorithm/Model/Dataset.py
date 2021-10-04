
class Feret_database_triplets(Dataset):
    def __init__(self,csv_file, tranformation=True):
        self.transformation=tranformation
        self.csv_file=pd.read_csv(csv_file)
        
        
    
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES=True

        indexer=self.csv_file.index
        label=self.csv_file.iloc[index,1]
        pose=self.csv_file.iloc[index,2]
        
        condition=self.csv_file['subject']==label
        pose_condition=self.csv_file['pose']==pose
        inv_condition=np.invert(condition)
        Neg_condition=pose_condition & inv_condition


        Pos_indices=indexer[condition].to_list()
        Neg_indices=indexer[Neg_condition].to_list()

        row_num_P=random.choice(Pos_indices)
        row_num_N=random.choice(Neg_indices)
        while row_num_P==index:
            row_num_P=random.choice(Pos_indices)

        img_name=self.csv_file.iloc[index,0]
        Pos_img_name=self.csv_file.iloc[row_num_P,0]
        Neg_img_name=self.csv_file.iloc[row_num_N,0]

        anchor_img=Image.open(img_name)
        Pos_img=Image.open(Pos_img_name)
        Neg_img=Image.open(Neg_img_name)
        images=[anchor_img,Pos_img,Neg_img]


        if self.transformation:
            num=random.random()
            if num>=0.5:
                anchor_img=F.hflip(anchor_img)
                Neg_img=F.hflip(Neg_img)
            
            else:
                Pos_img=F.hflip(Pos_img)
        
        
        transform=transforms.Compose([ToTensor(),Resize(size=(256,171)),RandomCrop(size=(256,171),padding=(40,26),padding_mode='reflect')])

        for i,image in enumerate(images):
            images[i]=transform(image) 
        
        
        
        return images
        


