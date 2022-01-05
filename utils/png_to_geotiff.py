
from osgeo import gdal
from osgeo.gdalconst import *

import json
import numpy as np


geotransforms = json.load(open("xview_geotransforms.json", "r"))
geomatrix = geotransforms["hurricane-michael_00000202_post_disaster.png"][0]
projection = geotransforms["hurricane-michael_00000202_post_disaster.png"][1]

inDs = gdal.Open("hurricane-michael_00000202_post_disaster.png")

rows = inDs.RasterYSize
cols = inDs.RasterXSize

outDs = gdal.GetDriverByName('GTiff').Create("hurricane-michael_00000202_post_disaster.tif", rows, cols, 3, GDT_Int16)

for i in range(1,4):
    outBand = outDs.GetRasterBand(i)
    outData = np.array(inDs.GetRasterBand(i).ReadAsArray())
    outBand.WriteArray(outData, 0, 0)
    outBand.FlushCache()
    outBand.SetNoDataValue(-99)

outDs.SetGeoTransform(geomatrix)
outDs.SetProjection(projection)

