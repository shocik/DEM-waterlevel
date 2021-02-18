import arcpy
from helper import *
from random import random
import numpy as np
"""
arcpy.env.workspace = r"D:\Doktorat\Badania\DEM-waterlevel\arc_script\output"
arcpy.CheckOutExtension("Spatial")

DEM_pth = r"J:\Pomiary\Kocinka 19-12-2020\Czestochowa - PCwiakala\Agisoft\dem2pl.tif"
ortho_pth = r"J:\Pomiary\Kocinka 19-12-2020\Czestochowa - PCwiakala\Agisoft\ortho2pl.tif"
points_csv_pth = r"D:\Doktorat\Badania\DEM-waterlevel\DoInterpolacji2.tab"
centerline_pth = r"J:\Pomiary\Kocinka 19-12-2020\Arc\nowy\2\centerline\centerline.shp"
tmp_path = create_dir(r"D:\Doktorat\Badania\DEM-waterlevel\arc_script\tmp")
dataset_path = create_dir(r"D:\Doktorat\Badania\DEM-waterlevel\arc_script\dataset")
coord_system = arcpy.Describe(DEM_pth).spatialReference#"ETRS_1989_Poland_CS2000_Zone_6"
extent = arcpy.Describe(DEM_pth).extent
cell_size = arcpy.Describe(ortho_pth).children[0].meanCellHeight
arcpy.env.extent = extent
arcpy.MakeXYEventLayer_management(points_csv_pth,"y","x", "points", coord_system, "z")
arcpy.Idw_3d("points", "z", os.path.join(tmp_path,"idw.tif"), cell_size=0.1, power=0.5)
arcpy.MosaicToNewRaster_management(DEM_pth+";"+os.path.join(tmp_path,"idw.tif"), tmp_path, "\denoised.tif", coord_system, "32_BIT_FLOAT", cell_size, "1", "MAXIMUM","FIRST")
arcpy.Buffer_analysis(centerline_pth, os.path.join(tmp_path,"buffer"), 4)
arcpy.CreateFishnet_management(os.path.join(tmp_path,"fishnet.shp"), origin_coord="{} {}".format(extent.XMin,extent.YMin), y_axis_coord="{} {}".format(extent.XMin,extent.YMax), cell_width=10, cell_height=10, labels="NO_LABELS", template=extent, geometry_type="POLYGON")
arcpy.MakeFeatureLayer_management (os.path.join(tmp_path,"fishnet.shp"), "fishnet")
arcpy.SelectLayerByLocation_management("fishnet", "INTERSECT", os.path.join(tmp_path,"buffer.shp"), selection_type="NEW_SELECTION", invert_spatial_relationship="INVERT")
arcpy.DeleteFeatures_management("fishnet")

divided_x_dem_pth = create_dir(os.path.join(tmp_path,"x_dem"))
divided_x_ort_pth = create_dir(os.path.join(tmp_path,"x_ort"))
divided_y_dem_pth = create_dir(os.path.join(tmp_path,"y_dem"))
#arcpy.FeatureClassToShapefile_conversion("fishnet", r"D:\Doktorat\Badania\DEM-waterlevel\arc_script\output\fishnet_del")
arcpy.SplitRaster_management(DEM_pth, divided_x_dem_pth, "_", "POLYGON_FEATURES", "TIFF", "BILINEAR", split_polygon_feature_class="fishnet")
arcpy.SplitRaster_management(ortho_pth, divided_x_ort_pth, "_", "POLYGON_FEATURES", "TIFF", "BILINEAR", split_polygon_feature_class="fishnet")
arcpy.SplitRaster_management(os.path.join(tmp_path,"denoised.tif"), divided_y_dem_pth, "_", "POLYGON_FEATURES", "TIFF", "BILINEAR", split_polygon_feature_class="fishnet")

"""
arcpy.CheckOutExtension("Spatial")
tmp_path = r"D:\Doktorat\Badania\DEM-waterlevel\arc_script\tmp"
types = ["x_dem","x_ort","y_dem"]
divided_pth = dict()
for type in types:
    divided_pth[type] = os.path.join(tmp_path,type)
dataset_path = create_dir(r"D:\Doktorat\Badania\DEM-waterlevel\arc_script\dataset")
train_dir = create_dir(os.path.join(dataset_path,"train"))
test_dir = create_dir(os.path.join(dataset_path,"test"))


for type in types:
    create_dir(os.path.join(train_dir,type))
    create_dir(os.path.join(test_dir,type))

arrays = dict()
for name in os.listdir(divided_pth["x_ort"]):
    if name.endswith(".TIF"):
        destination_pth = train_dir if random()>0.2 else test_dir
        new_name = name.replace("_","").replace("TIF","npy")
        for type in types:
            band_arrays = []
            arcpy.MakeRasterLayer_management(os.path.join(divided_pth[type],name), type)
            arr = arcpy.RasterToNumPyArray(type)
            #for band in range(arcpy.Describe(type).bandCount):
            #    arcpy.MakeRasterLayer_management(type, "band_raster", '', envelope="", band_index=str(band))
            #    arcpy.sa.Con(arcpy.sa.IsNull("band_raster"),
            #                    arcpy.sa.FocalStatistics("band_raster", arcpy.sa.NbrRectangle(3, 3),'MEAN'),
            #                    "band_raster")
            #    band_arrays.append(arcpy.RasterToNumPyArray("band_raster"))
            if arr.shape[0]==4:
                arr=arr[:3,1:-1,1:-1]
            else:
                arr = arr[1:-1,1:-1]
            
            np.save(os.path.join(destination_pth,type,new_name),arr)
