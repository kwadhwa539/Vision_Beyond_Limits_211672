import json
from shapely import wkt
from shapely.geometry import Polygon
import numpy as np 
from cv2 import fillPoly, imwrite

def open_json(json_file_path):
    """
    :param json_file_path: path to open inference json file
    :returns: the json data dictionary of localized polygon and their classifications 
    """

    with open(json_file_path) as jf:
        json_data = json.load(jf)
        inference_data = json_data['features']['xy']
        return inference_data

def create_image(inference_data):
    """
    :params inference_data: json data dictionary of localized polygon and their classifications
    :returns: an numpy array of 8-bit grey scale image with polygons filled in according to the key provided
    """

    damage_key = {'un-classified': (255,255,255), 'no-damage': (0,255,0), 'minor-damage': (255,0,0), 'major-damage': (0,165,255), 'destroyed': (0,0,255)}

    mask_img = np.zeros((1024,1024,3), np.uint8)

    for poly in inference_data:
        damage = poly['properties']['subtype']
        coords = wkt.loads(poly['wkt'])
        poly_np = np.array(coords.exterior.coords, np.int32)
        
        fillPoly(mask_img, [poly_np], damage_key[damage])
    
    return mask_img

def save_image(polygons, output_path):
    """
    :param polygons: np array with filled in polygons from create_image()
    :param output_path: path to save the final output inference image
    """

    # Output the filled in polygons to an image file
    imwrite(output_path, polygons)
  
def create_inference_image(json_input_path, image_output_path):
    """
    :param json_input_path: Path to output inference json file
    :param image_outut_pat: Path to save the final inference image
    """

    # Getting the inference data from the localization and classification 
    inference_data = open_json(json_input_path)

    # Filling in the polygons and readying the image format 
    polygon_array = create_image(inference_data)

    # Saving the image to the desired location
    save_image(polygon_array, image_output_path)

if __name__ == '__main__': 
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=
        """inference_image_output.py: Takes the inference localization and classification final outputs in json from and outputs an image ready to be scored based off the challenge parameters""")
    parser.add_argument('--input',
                        required=True,
                        metavar='/path/to/final/inference.json',
                        help="Full path to the final inference json")
    parser.add_argument('--output',
                        required=True,
                        metavar='/path/to/inference.png',
                        help="Full path to save the image to")

    args = parser.parse_args()

    # Creating the scoring image
    create_inference_image(args.input, args.output)
