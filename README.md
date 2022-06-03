# Fonts-classifier

def get_angle(img): 
    # RGB to gray 
    img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY) 
 
    # Gray to binary 
    th_box = int(img_gray.shape[0] * 0.007) * 2 + 1 
    img_bin_ = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, th_box, th_box) 
 
    img_bin = img_bin_ 
    num_rows, num_cols = img_bin.shape[:2] 
 
    best_zero, best_angle = None, 0 
    # iteratively rotate the image by half a degree 
    for my_angle in range(-20, 21, 1): 
        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows /2 ), my_angle/2, 1) 
        img_rotation = cv2.warpAffine(img_bin, rotation_matrix, (num_cols*2, num_rows*2), 
                                      borderMode=cv2.BORDER_CONSTANT, 
                                      borderValue=255) 
 
        img_01 = np.where(img_rotation > 127, 0, 1) 
        sum_y = np.sum(img_01, axis=1) 
        th_ = int(img_bin_.shape[0]*0.005) 
        sum_y = np.where(sum_y < th_, 0, sum_y) 
 
        num_zeros = sum_y.shape[0] - np.count_nonzero(sum_y) 
 
        if best_zero is None: 
            best_zero = num_zeros 
            best_angle = my_angle 
 
        # best rotation 
        if num_zeros > best_zero: 
            best_zero = num_zeros 
            best_angle = my_angle 
 
    return best_angle * 0.5
