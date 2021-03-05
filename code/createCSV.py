import face.modules as modules
from face.metaData import img_folder_path, cartoon_predictor, cartoon_expected_dis, cartoon_expected_areas
from hair import evaluate
import numpy as np


def extractPointsHair(batchPaths):
    batchImgPaths = []
    batchImgFaceFeatures = []
    batchImgHair = []

    for imagePath in batchPaths:
        shape_points = modules.predictPoints(imagePath, cartoon_predictor)
        if(len(shape_points) != 0):
            faceFeatures = modules.makeVector(
                shape_points, cartoon_expected_dis, cartoon_expected_areas)
            hair_output = evaluate.getHairOutput(imagePath)
            batchImgPaths.append(imagePath)
            batchImgHair.append(hair_output)
            batchImgFaceFeatures.append(faceFeatures)

    return batchImgPaths, batchImgHair, batchImgFaceFeatures


def main2():
    img_path_list = modules.getImagePaths(img_folder_path)

    csv_list = []
    # i = 608
    for (b, i) in enumerate(range(0, len(img_path_list), 32)):
        batchPaths = img_path_list[i:i + 32]
        batchImgPaths, batchImgHair, batchImgFaceFeatures = extractPointsHair(
            batchPaths)

        batchImgHair = np.vstack(batchImgHair)

        hairFeatures = evaluate.getHairVector(batchImgHair)
        for i in range(len(batchImgPaths)):
            temp = []
            temp.append(batchImgFaceFeatures[i])
            temp.append(list(hairFeatures[i]))
            temp.append(batchImgPaths[i])
            csv_list.append(temp)

    modules.makeCsv(csv_list)


if __name__ == '__main__':
    main2()
