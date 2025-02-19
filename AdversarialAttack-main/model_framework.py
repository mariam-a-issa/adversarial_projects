import sys
import torch

if __name__ == '__main__':
    _, model_type, dataset, attack, _ = [str(arg) for arg in sys.argv]
    print(f'model_type: {model_type}, dataset: {dataset}, attack: {attack}')

    # Preprocess images
    preprocessed_images = #TODO: get image data
    attack_data = #TODO get image data perturbed by attack model type

    model = build_model(model_type)

    # TODO: Get accuracy

def build_model(model_type):
    match (model_type):
        case "neurosymbolic":
            print("Building NeuroSymbolic Model...")
            model = extract_features(preprocessed_images)
            print("NeuroSymbolic Model - COMPLETE")
        case "hdc":
            print("Building HDC Model...")
            # TODO
            print("HDC Model - COMPLETE")

        case "dnn":
            print("Building DNN Model...")
            #TODO
            print("DNN Model - COMPLETE")
    return model


# TODO: do I want to share models or instantiate a new model for each database?
# TODO: need to vectorize?
def extract_features(image_data):

    # Instantiate the DNN for (1)shape extraction
    shape_dnn =

    # Instantiate the DNN for (2)color extraction
    color_dnn =

    # Instantiate the DNN for (3)texture extraction
    texture_dnn =

    # Get embeddings for each feature
    shape_embeddings = shape_dnn(image_data)
    color_embeddings = color_dnn(image_data)
    texture_embeddings = texture_dnn(image_data)

    binded_features = bind(bind(shape_embeddings, color_embeddings), texture_embeddings)
    return binded_features

#Bind two hypervectors together
def bind(hypervector_1, hypervector_2):
    binded_hypervectors =
    return binded_hypervectors