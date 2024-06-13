# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install ultralytics


from ultralytics import YOLO
import os
import numpy as np
import cv2 as cv


def xywh2xyxy_converter(xywh_bbox):
    # Convert bounding box's parameters from xywh type to xyxy type
    #   INPUT:
    #   xywh_bbox - bounding box's parameters of xywh type
    #   OUTPUT:
    #   xyxy_bbox - bounding box's parameters of xyxy type

    # Declate empty bounding box's parameters
    xyxy_bbox = np.zeros(4)

    # Mapping from x, y, w and h coordinates to x1, y1, x2, y2 coordinates
    xyxy_bbox[0] = xywh_bbox[0]-(xywh_bbox[2]/2.0)  # x1
    xyxy_bbox[2] = xywh_bbox[0]+(xywh_bbox[2]/2.0)  # y1
    xyxy_bbox[1] = xywh_bbox[1]-(xywh_bbox[3]/2.0)  # x2
    xyxy_bbox[3] = xywh_bbox[1]+(xywh_bbox[3]/2.0)  # y2

    return xyxy_bbox

def IoU(pred_bbox_xywh, act_bbox_xywh):
    # Compute IoU between prediction and ground truth
    #   INPUTS:
    #   pred_bbox_xywh - bounding box's paramterers from prediction in xywh type
    #   act_bbox_xywh - bounding box's paramterers from ground truth in xywh type
    #   OUTPUT:
    #   IoU - Intersection over union (IoU) between predicted and ground truth bounding boxes

    # Declate empty bounding box's parameters
    int_bbox = np.zeros(4)

    # Mapping from x, y, w and h coordinates to x1, y1, x2, y2 coordinates for prediction and ground truth
    pred_bbox = xywh2xyxy_converter(pred_bbox_xywh)
    act_bbox = xywh2xyxy_converter(act_bbox_xywh)

    # Calculation of intersected area's coordinates
    int_bbox[0] = max(pred_bbox[0], act_bbox[0])    # x1
    int_bbox[1] = max(pred_bbox[1], act_bbox[1])    # y1
    int_bbox[2] = min(pred_bbox[2], act_bbox[2])    # x2
    int_bbox[3] = min(pred_bbox[3], act_bbox[3])    # y2

    # Calculation of an intersected area
    int_area = max(int_bbox[2]-int_bbox[0],0)*max(int_bbox[3]-int_bbox[1],0)

    # Calculation of predicted bounding box's  area
    pred_area = (pred_bbox[2]-pred_bbox[0])*(pred_bbox[3]-pred_bbox[1])

    # Calculation of grounding true bounding box's  area
    act_area = (act_bbox[2]-act_bbox[0])*(act_bbox[3]-act_bbox[1])

    # Calculation of an union area
    union_area = pred_area + act_area - int_area

    # Calculation of IoU
    IoU = int_area/union_area

    return IoU



# Load model
model = YOLO('pedestrian_best.pt')  # pretrained YOLOv8n model


if __name__ == '__main__':

    # Declare class of the images' label you are looking for
    #   (Should be a string type)
    dst_cls = '0'

    # Paths of dataset in Yolo format
    dir_images = "E:\StudyTime\myProjects\ComputerVision/tmp3\mini_dataset/valid\images"    # Images
    dir_labels = "E:\StudyTime\myProjects\ComputerVision/tmp3\mini_dataset/valid\labels"    # Labels


    # Path for results
    dir_TP = 'E:\StudyTime\myProjects\ComputerVision/tmp3/results\True_Positive'    # True Positive results
    dir_TN = 'E:\StudyTime\myProjects\ComputerVision/tmp3/results\True_Negative'    # True Negative results
    dir_FP = 'E:\StudyTime\myProjects\ComputerVision/tmp3/results\False_Positive'   # False Positive results
    dir_FN = 'E:\StudyTime\myProjects\ComputerVision/tmp3/results\False_Negative'   # False Negative results

    # Bounding box
    pred_bbox = np.zeros(4)     # Bounding box used for predicted samples
    act_bbox = np.zeros(4)      # Bounding box used for samples from ground truth

    # IoU threshold
    IoU_threshold = 0.5

    # Iterate over files in the directory of images
    for file_im in os.listdir(dir_images):

        # Flags for classification of predicted images
        flag_TP = False     # Flag for True Positive Results
        flag_TN = False     # Flag for True Negative Results
        flag_FP = False     # Flag for False Positive Results
        flag_FN = False     # Flag for False Negative Results
        flag_dupl = False

        # Path for each image
        f1 = os.path.join(dir_images, file_im)

        # Checking if it is a file
        if os.path.isfile(f1):

            # Make prediction in the image
            result = model(f1, device= 0)[0]


            # Extracting path for labels
            filename = file_im.rstrip('.png')       # Remove .png from image's path name
            filename = filename.rstrip('.jpg')      # Remove .jpg from image's path name
            file_lab = filename + '.txt'            # Creating .txt path for labels

            # Path for the image's label file
            f2 = os.path.join(dir_labels, file_lab)

            # Read label file
            txt_file = open(f2, "r")
            lines = txt_file.read().splitlines()

            # Counting number of bounding boxes in prediction and ground truth
            pred_n_inst = result.boxes.xywhn.size(dim=0)    # Number of bounding boxes in prediction
            act_n_inst = len(lines)                         # # Number of bounding boxes in ground truth

            # Declaring number of matched bounding boxes between prediction and ground truth
            match_n_inst = 0

            # Declaring true positive masks for prediction and ground truth
            #   TP mask has i element equal to True if bounding box of i element's IoU > threshold
            pred_TP_mask = np.zeros(pred_n_inst, dtype = bool)  # True positive mask for prediction
            act_TP_mask = np.zeros(act_n_inst, dtype=bool)      # True positive mask for ground truth


            # Classify image for TP, TN, FP, FN class and fill TP mask
            if pred_n_inst == 0 and act_n_inst == 0:
                # If number of bounding boxes in both prediction and ground truth is equal to zero, then it is False Negative

                # Mark TN flag as True
                flag_TN = True

            else:

                # Iterate over bounding boxes in ground truth
                for i, line in enumerate(lines):

                    # Extract parameters from labels
                    value = line.split()

                    # Extract class from label
                    cls = value[0]

                    # Extract bounding box parameters from label
                    act_bbox[0] = float(value[1])
                    act_bbox[1] = float(value[2])
                    act_bbox[2] = float(value[3])
                    act_bbox[3] = float(value[4])


                    # If it is not desired class, then skip
                    if cls == dst_cls:

                        # Iterate over bounding boxes in prediction
                        for j, xywhn in enumerate(result.boxes.xywhn):

                            # Extract bounding box from prediction
                            x, y, w, h = xywhn
                            pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3] = float(x), float(y), float(w), float(h)

                            # Calculate IoU between prediction and ground truth
                            IoU_val = IoU(pred_bbox, act_bbox)

                            # Compare if IoU is satisfed
                            if IoU_val > IoU_threshold:

                                # Count matched IoU cases
                                match_n_inst = match_n_inst+1

                                # Mark TP masks between prediction and ground truth
                                act_TP_mask[i] = True
                                pred_TP_mask[j] = True

                    else:
                        continue


                act_n_True = act_TP_mask.sum()
                pred_n_True = pred_TP_mask.sum()

                if act_n_True > 0:
                    flag_TP = True

                if act_n_inst > act_n_True:
                    flag_FN = True

                if pred_n_inst > pred_n_True:
                    flag_FP = True


            # Declare image instances
            act_img = cv.imread(f1)
            act_img = cv.resize(act_img, (640, 640))
            pred_img = cv.resize(act_img, (640, 640))

            # Mark predicted bounding boxes in an image
            # Iterate over bounding boxes in ground truth
            for i, xywhn in enumerate(result.boxes.xywhn):

                # Extract grounding boxes from ground truth
                x, y, w, h = xywhn
                pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3] = float(x), float(y), float(w), float(h)

                # Declare bounding box's color for FN instance
                bbox_color = (0, 204, 0) # Color = green

                # Check if the instance's class is TP, then change the bounding box's color to red
                if pred_TP_mask[i] == True:

                    bbox_color = (0, 0, 204)  # Color = red

                # Convert bounding box's parameters from xywh type to xyxy type
                xyxy_pred_bbox = xywh2xyxy_converter(pred_bbox)

                # Add bounding box in the image
                pred_img = cv.rectangle(pred_img, (int(xyxy_pred_bbox[0]*640),int(xyxy_pred_bbox[1]*640)),
                                              ((int(xyxy_pred_bbox[2]*640),int(xyxy_pred_bbox[3]*640))), bbox_color, 5)


            # Mark bounding boxes from prediction to an image
            # Iterate over bounding boxes in prediction
            for i, line in enumerate(lines):

                # Extract parameters from prediction
                value = line.split()

                # Extract class from prediction
                cls = value[0]

                # Extract bounding box from prediction
                act_bbox[0] = float(value[1])
                act_bbox[1] = float(value[2])
                act_bbox[2] = float(value[3])
                act_bbox[3] = float(value[4])

                # Declare bounding box's color for FP instance
                bbox_color = (204, 0, 0)

                # If it is not desired class, then skip
                if cls == dst_cls:

                    # Check if the instance's class is TP, then change the bounding box's color to red
                    if act_TP_mask[i] == True:
                        bbox_color = (0, 0, 204)

                    # Convert bounding box's parameters from xywh type to xyxy type
                    xyxy_act_bbox = xywh2xyxy_converter(act_bbox)

                    # Add bounding box in the image
                    act_img = cv.rectangle(act_img, (int(xyxy_act_bbox[0] * 640), int(xyxy_act_bbox[1] * 640)),
                                            ((int(xyxy_act_bbox[2] * 640), int(xyxy_act_bbox[3] * 640))), bbox_color, 5)

            # Declare origin for the second line in y-axis for the image
            pred_orig_y_coord = 50

            # Add text for the predicted image
            pred_img = cv.putText(pred_img, "Predicted:", (0,25), cv.FONT_HERSHEY_SIMPLEX,
                                  1, (0, 0, 0),2)

            # Check if prediction is TN
            if flag_TN == True:

                # Add text for predicted TN image
                pred_img = cv.putText(pred_img, " TN", (0, pred_orig_y_coord),
                                      cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            else:

                if flag_TP == True:

                    # Add TP count text for the predicted image
                    pred_img = cv.putText(pred_img, " TP (+dupl.) #:"+str(pred_n_True), (0, pred_orig_y_coord),
                                          cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 204), 2)

                    # Change origin of the next text in y-axis
                    pred_orig_y_coord = pred_orig_y_coord+25

                # Check if FP instance exists
                if flag_FP == True:

                    # Add FP count for the predicted image
                    pred_img = cv.putText(pred_img, " FP #:" + str(pred_n_inst-pred_n_True), (0, pred_orig_y_coord),
                                          cv.FONT_HERSHEY_SIMPLEX, 1, (0, 204, 0), 2)



            # Declare origin for the second line in y-axis for the image
            act_orig_y_coord = 50

            # Add text for the ground truth image
            act_img = cv.putText(act_img, "Ground Truth:", (0,25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),2)

            # Check if prediction is TN
            if flag_TN == True:



                # Add TN text for the ground truth image
                act_img = cv.putText(act_img, " TN", (0, act_orig_y_coord),
                                     cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            else:

                if flag_TP == True:

                    # Add TP count text for the ground truth image
                    act_img = cv.putText(act_img, " TP #:" + str(act_n_True), (0, act_orig_y_coord),
                                         cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 204), 2)

                    # Change origin of the next text in y-axis
                    act_orig_y_coord = act_orig_y_coord+25


                # Check if FN instance exists
                if flag_FN == True:

                    # Add FP count for the predicted image
                    act_img = cv.putText(act_img, " FN #:" + str(act_n_inst-act_n_True), (0, act_orig_y_coord),
                                         cv.FONT_HERSHEY_SIMPLEX, 1, (204, 0, 0), 2)

            # Conactenate both predicted and ground truth image
            anl_img = cv.hconcat([pred_img, act_img])



            # Save the resultant image in according TP, FP, FN, TN folders
            # Check if the resultant image has TP instance
            if flag_TP == True:

                # Save in the resultant image in TP folder
                cv.imwrite(os.path.join(dir_TP, filename)+'.jpg', anl_img)

            # Check if the resultant image has TN instance
            if flag_TN == True:

                # Save in the resultant image in TN folder
                cv.imwrite(os.path.join(dir_TN, filename)+'.jpg', anl_img)

            # Check if the resultant image has FP instance
            if flag_FP == True:

                # Save in the resultant image in FP folder
                cv.imwrite(os.path.join(dir_FP, filename)+'.jpg', anl_img)

            # Check if the resultant image has FN instance
            if flag_FN == True:

                # Save in the resultant image in FN folder
                cv.imwrite(os.path.join(dir_FN, filename)+'.jpg', anl_img)


            print("act_n_True: ", str(act_n_True))
            print("pred_n_True: ", str(pred_n_True))




