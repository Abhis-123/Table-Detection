import numpy as np

MIN_WIDTH_BOX=5
MIN_HEIGHT_BOX=5
def sanitize_coord(coordinates, width, height):
    """
    points are: [[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]]
    it sanitize the coordinates that are extracted from a xml file. Valid for this dataset,
    to be updated in case the dataset changes
    Returning as dict: xmin, ymin, xmax, ymax
    :param coordinates:[[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]]
    :return: dict with xmin, ymin, xmax, ymax coordinates
    """
    coordinates = coordinates.split()
    points = []
    for point in coordinates:
        point = point.split(',')
        points.append(point)
    new_points = {
        'xmin': points[0][0],
        'ymin': points[0][1],
        'xmax': points[3][0],
        'ymax': points[3][1]
    }
    # logger.info(new_points)
    # logger.info('width: {w}, height: {h}'.format(w=width, h=height))
    # check if coords are inverted
    if int(new_points['ymin']) > int(new_points['ymax']):
        temp = int(new_points['ymin'])
        new_points['ymin'] = int(new_points['ymax'])
        new_points['ymax'] = temp
    if int(new_points['xmin']) > int(new_points['xmax']):
        temp = new_points['xmin']
        new_points['xmin'] = int(new_points['xmax'])
        new_points['xmax'] = temp
    if int(new_points['ymin']) < 0:
        new_points['ymin'] = 0
    if int(new_points['xmin']) < 0:
        new_points['xmin'] = 0
    if int(new_points['ymax']) > height:
        new_points['ymax'] = height
    if int(new_points['xmax']) > width:
        new_points['xmax'] = width
    if (int(new_points['xmax']) - int(new_points['xmin'])) < MIN_WIDTH_BOX or \
            (int(new_points['ymax']) - int(new_points['ymin'])) < MIN_HEIGHT_BOX:
        new_points = None
    return new_points





def voc_to_yolo(image_height, image_width, bboxes):
    """
    voc  => [x1, y1, x2, y2]
    yolo => [xmid, ymid, w, h] (normalized)
    """
    
    bboxes = bboxes.astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]]/ image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]]/ image_height
    
    w = bboxes[..., 2] - bboxes[..., 0]
    h = bboxes[..., 3] - bboxes[..., 1]
    
    bboxes[..., 0] = bboxes[..., 0] + w/2
    bboxes[..., 1] = bboxes[..., 1] + h/2
    bboxes[..., 2] = w
    bboxes[..., 3] = h
    
    return bboxes

def yolo_to_voc(image_height, image_width, bboxes):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    voc  => [x1, y1, x2, y1]
    
    """ 
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    bboxes[..., [0, 2]] = bboxes[..., [0, 2]]* image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]]* image_height
    
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]]/2
    bboxes[..., [2, 3]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]
    
    return bboxes




def norm_bbox(bboxes, image_height=720, image_width=1280):
    """
    coco => [xmin, ymin, w, h]
    yolo => [xmid, ymid, w, h] (normalized)
    """
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    # normolizinig
    bboxes[..., [0, 2]]= bboxes[..., [0, 2]]/ image_width
    bboxes[..., [1, 3]]= bboxes[..., [1, 3]]/ image_height
    
    # converstion (xmin, ymin) => (xmid, ymid)
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]/2
    
    return bboxes


def denorm_bbox(bboxes, image_height=720, image_width=1280):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    coco => [xmin, ymin, w, h]
    
    """ 
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int
    
    # denormalizing
    bboxes[..., [0, 2]]= bboxes[..., [0, 2]]* image_width
    bboxes[..., [1, 3]]= bboxes[..., [1, 3]]* image_height
    
    # converstion (xmid, ymid) => (xmin, ymin) 
    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]]/2
    
    return bboxes

# Scale the bounding boxes according to the size of the resized image. 
def scale_bbox(orignal_image_width, orignal_image_height,new_image_width,new_image_height, bboxes):
    # Get scaling factor
    scale_x = orignal_image_width/new_image_width
    scale_y = orignal_image_height/new_image_height
    
    scaled_bboxes = []
    for bbox in bboxes:
        x = int(np.round(bbox[0]*scale_x, 4))
        y = int(np.round(bbox[1]*scale_y, 4))
        x1 = int(np.round(bbox[2]*(scale_x), 4))
        y1= int(np.round(bbox[3]*scale_y, 4))

        scaled_bboxes.append([x, y, x1, y1]) # xmin, ymin, xmax, ymax
        
    return scaled_bboxes

