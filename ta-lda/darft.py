import glob

data_dir = './data'


file_names = glob.glob(data_dir + "/*/*.csv")
file_names = glob.glob(data_dir + "/*/*.csv")

for f in file_names:
    print(f)
