import os 
import shutil
import math

# print(os.listdir(os.path.join(os.getcwd(), "datasets/Train/view_1/View1_photo")))

xc = os.listdir(os.path.join(os.getcwd(), "../../../storage/research/mview/aerial_2/tower"))
yc = os.listdir(os.path.join(os.getcwd(), "../../../storage/research/mview/ground_2/tower"))
xc = sorted(xc)
yc = sorted(yc)
# print(type(xc))
# print(type(yc))
# d = [x for x in xc]
# print(type(d))

count = 0
len_xc = len(xc)
print("length before split", len_xc)
print("length before split", len(yc))
print("expected train length after split", math.ceil(len_xc - (0.25 * len_xc)))
print("expected test length after split", math.ceil(0.25 * len_xc))

check = math.ceil(len_xc - (0.25 * len_xc))
# print("---")
# for ii in yc:
# 	print(ii)
	# break
for idx, (im1, im2) in enumerate(zip(xc, yc)):
	# print(im1)
	# print(im2)
	# break
	# print("\n")
	# if idx == 10:
	# 	break
	if idx > check:
		shutil.move(os.path.join(os.path.join(os.getcwd(), "../../../storage/research/mview/aerial_2/tower"),im1), os.path.join(os.path.join(os.getcwd(), "../../../storage/research/mview/test/aerial/tower"), im1))
		shutil.move(os.path.join(os.path.join(os.getcwd(), "../../../storage/research/mview/ground_2/tower"),im2), os.path.join(os.path.join(os.getcwd(), "../../../storage/research/mview/test/ground/tower"),im2))
		# break;

	else:
		# print("jsa---", im)
		shutil.move(os.path.join(os.path.join(os.getcwd(), "../../../storage/research/mview/aerial_2/tower"),im1), os.path.join(os.path.join(os.getcwd(), "../../../storage/research/mview/train/aerial/tower"),im1))
		shutil.move(os.path.join(os.path.join(os.getcwd(), "../../../storage/research/mview/ground_2/tower"),im2), os.path.join(os.path.join(os.getcwd(), "../../../storage/research/mview/train/ground/tower"),im2))
		# break 

