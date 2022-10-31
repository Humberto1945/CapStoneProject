sclass_path = '/content/drive/MyDrive/Python/PascalVOC2012/VOC2012/SegmentationClass/'
jpeg_path = '/content/drive/MyDrive/Python/PascalVOC2012/VOC2012/JPEGImages/'


#using text files to load filenames into lists
def loadDatasets(filename, imgarray):
    f = open(filename)
    for line in iter(f):
        line = line[0:15]
        imgarray.append(line)