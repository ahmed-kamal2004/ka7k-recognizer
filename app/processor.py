import cv2 as cv
import numpy as np


class Processor:
    @staticmethod
    def process(img):
            gray_img = Processor.get_gray_img(img)

            table_img = Processor.get_table_img(gray_img)

            binary_table_img = Processor.binary_img(table_img)

            output__closed_img = Processor.close_img(binary_table_img)

            return output__closed_img,table_img,binary_table_img
        

    @staticmethod
    def show_image(img):
        resized_img = cv.resize(img,(1000,800), fx=0, fy=0, interpolation=cv.INTER_LINEAR)
        cv.imshow("Image",resized_img)

        cv.waitKey(5000)
        cv.destroyAllWindows()

    @staticmethod
    def binary_img(img,thres = 120):
        # show_image(img)

        _, binary = cv.threshold(img, thres, 255, cv.THRESH_BINARY_INV)

        # show_image(binary)
        return binary

    @staticmethod
    def apply_caney_edge_detector(img):
        smoothed_img = cv.GaussianBlur(img, (5, 5), 0)
        edges = cv.Canny(smoothed_img, threshold1= 30, threshold2= 70)

        return edges

    @staticmethod
    def get_gray_img(img):
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return gray_img

    @staticmethod
    def get_table_img(img):

        binary_image = Processor.binary_img(img)

        #  show_image(binary_image)

        contours, _ = cv.findContours(binary_image, cv.RETR_TREE,cv.CHAIN_APPROX_NONE)

        sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
        
        rectangular_contours = []

        for contour in sorted_contours:
            epsilon = 0.04 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                rectangular_contours.append(contour)

        chosen_one = None
        if cv.contourArea(rectangular_contours[0]) > 0.8 * img.shape[0] * img.shape[1]:  # Adjust threshold as necessary
            chosen_one = rectangular_contours[2]
        else:
            chosen_one = rectangular_contours[0]

        angle = abs((90 if cv.minAreaRect(chosen_one)[-1] > 45 else 0 )- cv.minAreaRect(chosen_one)[-1])

        rotated_image = Processor.rotate_image(img,angle)

        
        x, y, w, h = cv.boundingRect(chosen_one)

        table_roi = rotated_image[y:y+h, x:x+w]

        ## Calc the angle of shifting

        return table_roi

    @staticmethod
    def close_img(binary_table_img):

        kernel_vertical = np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ], dtype=np.uint8)

        kernel_horizon = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0]
        ], dtype=np.uint8)

        kernel = np.ones((3, 3), dtype=np.uint8)


        in_img = binary_table_img
        binary_table_img_horizon = cv.erode(
            in_img,kernel_horizon,iterations=10
        )
        binary_table_img_vertical = cv.erode(
            in_img,kernel_vertical,iterations=15
        )
        
        output_img = cv.max(binary_table_img_horizon,binary_table_img_vertical)

        output_img = cv.dilate(
            output_img,kernel,iterations=2
        )


        return output_img

    @staticmethod
    def rotate_image(image, angle):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        rotation_matrix = cv.getRotationMatrix2D(center, -angle, 1.0)
        
        rotated_image = cv.warpAffine(image, rotation_matrix, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
        
        return rotated_image
