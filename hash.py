
from imutils import paths
import cv2

def dhash(image, hashSize=8):
	# resize the input image, adding a single column (width) so we
	# can compute the horizontal gradient
	resized = cv2.resize(image, (hashSize + 1, hashSize))
	# compute the (relative) horizontal gradient between adjacent
	# column pixels
	diff = resized[:, 1:] > resized[:, :-1]
	# convert the difference image to a hash
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

needlePaths = list(paths.list_images("imgs"))

# loop over the needle paths
for p in needlePaths:
	# load the image from disk
	image = cv2.imread(p)
	# if the image is None then we could not load it from disk (so
	# skip it)
	if image is None:
		continue
	# convert the image to grayscale and compute the hash
	imageHash = dhash(image)
	print(imageHash)
	cv2.imwrite('for_copy/{}.jpg'.format(imageHash), image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
