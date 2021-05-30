import pixellib
from pixellib.semantic import semantic_segmentation
import os
from PIL import Image

segment_image = semantic_segmentation()
segment_image.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")


file = open("test.txt", 'r')
lis = file.readlines()
img_list = []
for i in lis:
    temp = i.replace("\n", "")
    img_list.append(temp)
print(img_list)

val = open("val.txt", 'r')
v = val.readlines()
val_list = []
for j in v:
    tmp = j.replace("\n", "")
    val_list.append(tmp)


for root, dirs, files in os.walk("dataset/images"):
    for file in files:
        file_name = file.replace(".jpg", "")
        if file_name in val_list:
            val_path = os.path.join(root, file)
            segment_image.segmentAsPascalvoc(val_path, output_image_name="val/"+file_name+".png")
            print(segment_image.segmentAsPascalvoc(val_path))

for root, dirs, files in os.walk("dataset/images"):
    for file in files:
        file_name = file.replace(".jpg", "")
        if file_name in img_list:
            val_path = os.path.join(root, file)
            segment_image.segmentAsPascalvoc(val_path, output_image_name="results/"+file_name+".png")
            print(segment_image.segmentAsPascalvoc(val_path))

for root, dirs, files in os.walk("val"):
    for file in files:
        val_image = Image.open(os.path.join(root, file))
        val_image = val_image.convert("P")
        val_image.save("val/"+file)

for root, dirs, files in os.walk("results"):
    for file in files:
        test_image = Image.open(os.path.join(root, file))
        test_image = test_image.convert("P")
        test_image.save("results/"+file)
