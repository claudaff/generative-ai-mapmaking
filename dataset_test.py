from loadDataset import MyDataset

dataset = MyDataset()
print(len(dataset))

item = dataset[44]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)
