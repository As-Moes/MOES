
import cv2

# Resize the requested image using a percentage factor (0-1)
def resize_by(image, scale_factor):
    width, height = image.shape[1], image.shape[0]
    new_width     = int(width * scale_factor)
    new_height    = int(height * scale_factor)
    dimensions    = (new_width, new_height)
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)

# Resize the requested image using a new size
def resize_to(image, new_size):
    new_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return new_image

# Show the requested image
def show(image, title='Image', scale=1.0, free=False):
    cv2.namedWindow(title)
    cv2.moveWindow(title, 40,30)
    image = resize_by(image, scale)
    cv2.imshow(title, image)
    wait_time = 1000
    while cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1:
        keyCode = cv2.waitKey(wait_time)
        if(keyCode & 0xFF) == ord("q") or free:
            break
    cv2.destroyWindow(title)
    
# Read the requested image 
def read(image_path, scale=1.0):
    image = cv2.imread(image_path)
    image = resize_by(image, scale) 
    return image

# Write the requested image 
def write(image, image_output_path, scale=1.0):
    image = resize_by(image, scale)
    cv2.imwrite(image_output_path, image)

