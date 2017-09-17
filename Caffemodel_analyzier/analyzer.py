#!/usr/bin/env python

"""analyzer.py: exam how your pretrained network act by testing new pictures."""

__author__ = "HaojieYuan"
__email__ = "haojie.d.yuan@gmail.com"
__version__ = "0.1"

import model
import transformer
import config
import image_and_label
import predictor


if __name__ == "__main__":

    image_list = []
    label_list = []
    predict_label_list = []

    # Read pretrained network
    _net = model.read(config.use_GPU, config.GPU_device, config.deploy_net,
                    config.Model_file, config.image_batch, config.reshape_channel,
                    config.reshape_height, config.reshape_width)
    
    # Set up image reshape transformer.
    _transformer = transformer.set(_net, config.Mean_file, config.bin2npy)
    
    # Get list file here, list file can be use again next time if you generate one.
    if config.use_raw_image == True:

        # Load list file config.
        list_config = {}
        list_config['_path'] = config.image_path
        list_config['class_number'] = config.class_number
        list_config['_label'] = config._label
        list_config['file_path'] = config.generated_list_file 
        list_config['amount'] = config.image_amount
        list_config['absolute_path'] = 1

        # Generate list file.
        list_file = image_and_label.build_list(list_config)
        
    else:
        list_file = config.list_file

    # Get image list and label list separately.
    image_list, label_list = image_and_label.get(list_file)

    # Get prediction of each image.
    for image in image_list:
        prediction = predictor.predict(image, _transformer, _net)
        predict_label_list.append(prediction)

    # For further analyze.
    class_right_count = []
    class_wrong_count = []

    # Set inintial value for each class count.
    for i in range(0, config.class_number):
        class_right_count.append(0)
        class_wrong_count.append(0)

    # Get statistical data.
    for predict_label, label in zip(predict_label_list, label_list):
        
        # Find index of label.
        _index = config._label.index(label)

        if predict_label == label:
            # Increase right count number.
            class_right_count[_index] = class_right_count[_index] + 1
        else:
            # Increase wrong count number.
            class_wrong_count[_index] = class_wrong_count[_index] + 1


    # Print Total right rate.
    right_rate = sum(class_right_count)/ (len(label_list) * 1.0)
    print "\nResult: Total right rate is %f ." % right_rate


    # Print each class's right rate.
    for i in range(0, config.class_number):

        total_count = class_right_count[i] + class_wrong_count[i]
        
        # Avoid situations that denominator is 0.
        if total_count == 0:
            print "No test sample of %s ." % config.class_name[i]
        else:
            right_rate = class_right_count[i] * 1.0 / total_count
            print "%s right rate is %f ." % (config.class_name[i], right_rate)
