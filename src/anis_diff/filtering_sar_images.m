clc; close all;

% Anis diff parametrs
DELTA_T = 0.25;
K = 0.05;
T = 13;

% Paths
SAR_IMAGES_PATH = "./sar_images/";
SAR_IMAGES_FILTERED_PATH = "./sar_images_filtered/";

% list of images in folder
FILES = dir("./sar_images/*.png");
LEN = length(FILES);

for i = 1:LEN
    img_name = FILES(i).name;

    % Open image and conver to grayscale
    original_img = mat2gray(im2gray(imread(SAR_IMAGES_PATH + img_name)));
    
    % Apply anis diff
    filtered_quad_img = AnisotropicDiffusionQuad(original_img, T, K, DELTA_T);
    
    % Save images
    imwrite(filtered_quad_img, SAR_IMAGES_FILTERED_PATH + img_name);
end
