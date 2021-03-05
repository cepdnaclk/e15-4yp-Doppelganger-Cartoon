# %%
import face.modules as modules
from gender.malFemalDetector import age_gender_detector
import hair.evaluate as evaluate
from face.metaData import human_predictor, human_expected_areas, human_expected_dis
import face.modules as md
from matplotlib import pyplot as plt
from keras.preprocessing.image import load_img
from operator import itemgetter
import numpy as np
import ast
import pandas


csv_results_set = pandas.read_csv(
    r'data7_new.csv', header=None)


def main():
    euclidean_dis_list = []
    w = 0.8
    imgPath = r'rapunzel_real.png'
    shape_points = md.predictPoints(imgPath, human_predictor)
    face_vector = md.makeVector(shape_points, human_expected_dis,
                                human_expected_areas)

    mask_output = evaluate.getHairOutput(imgPath)
    hair_vector = evaluate.getHairVector(mask_output)

    gender_output = age_gender_detector(imgPath)

    m = 0

    csv_results = csv_results_set[csv_results_set[3] == gender_output.lower()]
    print(csv_results.shape)

    for face, hair in zip(csv_results[0], csv_results[1]):
        faceVal = list(ast.literal_eval(face))
        hairVal = list(ast.literal_eval(hair))

        euclidean_distance_face = md.findEuclideanDistance(
            np.array(face_vector, dtype=float), np.array(faceVal, dtype=float))
        euclidean_distance_hair = md.findEuclideanDistance(
            np.array(hair_vector, dtype=float), np.array(hairVal, dtype=float)) * 10000
        euclidean_distance = w * euclidean_distance_face + \
            (1-w) * euclidean_distance_hair
        euclidean_dis_list.append(euclidean_distance)
        m = m+1
    indices, sorted_euclidean_dis = zip(
        *sorted(enumerate(euclidean_dis_list), key=itemgetter(1)))

    csv_results = list(csv_results[2])

    fig = plt.figure(1)
    img = load_img(imgPath)
    img = np.array(img)
    plt.subplot(1, 6, 1)
    plt.axis('off')
    plt.title('real', fontsize=8)
    plt.imshow(img)

    j = 2
    for i in range(5):
        print(indices[i])
        print(sorted_euclidean_dis[i], indices[i])
        print(csv_results[indices[i]])
        resulted_image = load_img(csv_results[indices[i]])
        resulted_image = np.array(resulted_image)

        plt.subplot(1, 6, j)
        plt.axis('off')
        j = j+1
        plt.title(i+1, fontsize=8)
        plt.imshow(resulted_image)

    plt.show()
    fig.savefig(
        r"rapunzel_match.png", dpi=190)
    plt.close()


if __name__ == '__main__':
    main()
