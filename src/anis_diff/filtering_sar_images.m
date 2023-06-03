clc; close all;

% Anis diff parametrs
DELTA_T = 0.25;
K = 0.05;
T = 13;

% Rayleigh distr parametr
SCALE = 0.2707;

% Paths
NOISED_PATH = "./noised/";
FILTERED_EXP_PATH = "./filtered_exp/";
FILTERED_QUAD_PATH = "./filtered_quad/";
ORIGINAL_PATH = "./original/";
GRAY_PATH = "./gray_images/";

% list of images in folder
FILES = dir("./original/*.jpg");
LEN = length(FILES);

ssim_quad = 0;
%ssim_exp = 0;
gmsd = 0;


for i = 1:LEN
    img_name = FILES(i).name;

    % Open image and conver to grayscale
    original_img = mat2gray(rgb2gray(imread(ORIGINAL_PATH + img_name)));
    
    % Add noise to image
    noised_img = add_noise(original_img, SCALE);
    
    % Apply anis diff
    filtered_exp_img = AnisotropicDiffusionExp(noised_img, T, K, DELTA_T);
    filtered_quad_img = AnisotropicDiffusionQuad(noised_img, T, K, DELTA_T);
     
    %figure('Name','Original'), imshow(original_img);
    %figure('Name','Filtered'), imshow(filtered_img);
    %figure('Name','Noised'), imshow(noised_img);
    
    % Check SSIM
    %ssim_value_exp = ssim(original_img, filtered_exp_img);
    ssim_value_quad = ssim(original_img, filtered_quad_img);
    gmsd_value = gmsdMetric(original_img, filtered_quad_img);

    ssim_quad = ssim_quad + ssim_value_quad;
    %ssim_exp = ssim_exp + ssim_value_exp;
    gmsd = gmsd + gmsd_value;
        
    data_log = num2str(img_name) + " ssim: " + num2str(ssim_value_quad) + " gmsd:" + num2str(gmsd_value);
    %ssim_sum = ssim_sum + ssim_value;
    disp(data_log);
    
    % Save images
    imwrite(noised_img, NOISED_PATH + img_name);
    imwrite(filtered_exp_img, FILTERED_EXP_PATH + img_name);
    imwrite(filtered_quad_img, FILTERED_QUAD_PATH + img_name);
    imwrite(original_img, GRAY_PATH + img_name);
end

% Average ssim

avg_metrics = " ssim avg: " + num2str(ssim_quad / LEN) + " gmsd avg: " + num2str(gmsd / LEN);
disp(avg_metrics)
