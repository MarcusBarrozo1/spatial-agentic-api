import geopandas as gpd
import numpy as np
import rasterio as rio
from rasterio.mask import mask

#Path to the vector and raster data
vector_data_path = 'sample_data/'
raster_data_path = 'sample_data/'

#Load the vector and raster data
vector = gpd.read_file(vector_data_path + 'sample_area.shp')
raster = rio.open(raster_data_path + 'B2_S2.tif')

#Get the CRS of the vector and raster data
vector_crs = vector.crs
raster_crs = raster.crs

#Verify if the CRS of the vector and raster data are the same
def verify_crs(vector_crs, raster_crs):
    if vector_crs != raster_crs:
        print("CRS do vetor:", vector_crs)
        print("CRS do raster:", raster_crs)
        print(f"Reprojetando vetor de {vector.crs} para {raster.crs}...")
        vector = vector.to_crs(raster.crs)
    else:
        print("O CRS do vetor e do raster são compatíveis.")

verify_crs(vector_crs=vector.crs, raster_crs=raster.crs)

print("CRS do vetor:", vector)
print("--------------------------------------\n")

#Check if the CRS of the vector and raster data are the same.
print("\n--- SANITY CHECK DE BOUNDING BOXES ---")
print(f"Limites do Vetor : {vector.total_bounds}  -> [minX, minY, maxX, maxY]")
print(f"Limites do Raster: {raster.bounds} -> [minX, minY, maxX, maxY]")
print("--------------------------------------\n")

#Get the geometry of the vector data
geometry = [geom for geom in vector.geometry]

#Use the geometry to mask the raster data
out_image, out_transform = mask(dataset=raster, shapes=geometry, crop=True)

print(f"Shape da matriz extraída: {out_image.shape}")
print(f"Valores únicos na matriz cortada: {np.unique(out_image)}")
print("--------------------------------------\n")

#Divide the matrix into 256x256 blocks
patch_size = 256

#Retrieve the cut matrix dimensions
_, height, width = out_image.shape

#Calculate the number of patches in the y and x directions
y = height // patch_size
x = width // patch_size

patch_list = []

#Iterate through the cut matrix and extract 256x256 blocks
for i in range(y):
    for j in range(x):
        y_start = i * patch_size
        y_end = (i + 1) * patch_size

        x_start = j * patch_size
        x_end = (j + 1) * patch_size

        patch = out_image[:, y_start:y_end, x_start:x_end]
        patch_list.append(patch)

#Convert the list of patches to a tensor
tensor_patches = np.array(patch_list)
print(f"Shape do tensor de blocos: {tensor_patches.shape}")

#Channels-last format for TensorFlow
tensor_transp = tensor_patches.transpose(0, 2, 3, 1)

#Reduce processing time converting to float32
tensor_float = tensor_transp.astype(np.float32)

#Normalize the values for TensorFlow
tensor_normalized = tensor_float / 10000.0
tensor_tf = tensor_normalized.clip(0.0, 1.0)

print(f"Shape para o TensorFlow: {tensor_tf.shape}")
print(f"Valor Máximo Normalizado: {np.max(tensor_tf)}")
print(f"Tipo de Dado: {tensor_tf.dtype}")